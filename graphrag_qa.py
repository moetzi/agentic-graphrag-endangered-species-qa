"""
Agentic GraphRAG over Neo4j with a ReAct loop driven by Ollama.

Tools are derived from `evaluation.tool_catalog.TOOL_CATALOG` so the agent's
runtime tools cannot drift from the ground truth's `expected_tools`.

One optional ablation component can be toggled at build time:

    build_app(with_validator=True)  # rule-based grounded-answer check
                                    # with one self-correction retry

Run a single question (CLI):
    python graphrag_qa.py "What threats does the Sumatran orangutan face?"
    python graphrag_qa.py --validator "..."

Programmatic entry point used by the evaluation runner:
    from graphrag_qa import build_app, ask
    app = build_app(with_validator=True)
    result = ask(app, "...question...")
    # result -> {"answer": str, "tool_calls": [...], "messages": [...],
    #            "validation": [...], "retry_count": int}
"""
from __future__ import annotations

import os
from typing import Annotated, Any

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field

# ---- LangChain / LangGraph imports ---------------------------------------- #
# Neo4jGraph still ships in langchain-community; the suggested replacement
# (`langchain-neo4j`) isn't installed in this project, so we silence the
# deprecation warning instead of pulling in another dependency.
import warnings
from langchain_core._api import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

from langchain_community.graphs import Neo4jGraph  # noqa: E402
from langchain_core.messages import (              # noqa: E402
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool    # noqa: E402
from langchain_ollama import ChatOllama            # noqa: E402
from langgraph.graph import StateGraph, START, END  # noqa: E402
from langgraph.graph.message import add_messages   # noqa: E402
from langgraph.prebuilt import ToolNode, tools_condition  # noqa: E402

from evaluation.tool_catalog import TOOL_CATALOG   # noqa: E402
from evaluation.validator import (                 # noqa: E402
    ValidationResult,
    validate as validate_answer,
    violations_message,
)

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
load_dotenv(find_dotenv())

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are an Expert Ecological Assistant answering questions about "
        "endangered species using a Neo4j knowledge graph. "
        "Always use the provided tools to retrieve facts. Do not guess or "
        "invent species, threats, habitats, or conservation actions. "
        "If a tool returns an empty list, state clearly that no data was "
        "found instead of fabricating an answer. "
        "Prefer the most specific tool that matches the user's question, and "
        "combine multiple tools for multi-hop questions when appropriate."
    )
)


# --------------------------------------------------------------------------- #
# State                                                                       #
# --------------------------------------------------------------------------- #
class AgentState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    # Validator metadata (None when the validator node is disabled).
    validation: ValidationResult | None = None
    retry_count: int = 0
    # Set by the validator when it has decided (and budgeted) a retry.
    # Read by the post-validate conditional edge to route back to the
    # agent. Reset to False after the routing decision is made.
    pending_retry: bool = False
    question: str = ""

    model_config = {"arbitrary_types_allowed": True}


# --------------------------------------------------------------------------- #
# Tool factory: build one StructuredTool per entry in TOOL_CATALOG.           #
# --------------------------------------------------------------------------- #
def _make_tool(name: str, spec: dict, graph: Neo4jGraph) -> StructuredTool:
    """Wrap a Cypher template as a parametric LangChain tool."""
    cypher: str = spec["cypher"]
    params: list[str] = spec["params"]
    description: str = spec["description"]

    def _run(**kwargs: Any) -> list[str] | str:
        # Defensive parameter check — the LLM occasionally hallucinates names.
        missing = [p for p in params if p not in kwargs]
        if missing:
            return f"Missing parameters: {missing}"
        try:
            rows = graph.query(cypher, params={p: kwargs[p] for p in params})
        except Exception as exc:  # pragma: no cover - surfaced to the agent
            return f"Cypher error: {exc}"
        if not rows:
            return []
        # Each tool returns a single labelled column; flatten to a list of
        # values for cleaner agent reasoning.
        first_key = next(iter(rows[0].keys()))
        return [r[first_key] for r in rows if r.get(first_key) is not None]

    # Build a Pydantic schema for the tool's arguments so Ollama gets a
    # well-formed JSON schema for tool calling.
    args_schema = type(
        f"{name}_Args",
        (BaseModel,),
        {
            "__annotations__": {p: str for p in params},
        },
    )

    return StructuredTool.from_function(
        func=_run,
        name=name,
        description=description,
        args_schema=args_schema,
    )


def build_tools(graph: Neo4jGraph) -> list[StructuredTool]:
    return [_make_tool(name, spec, graph) for name, spec in TOOL_CATALOG.items()]


# --------------------------------------------------------------------------- #
# Graph factory                                                               #
# --------------------------------------------------------------------------- #
def _connect_graph() -> Neo4jGraph:
    if not (NEO4J_USER and NEO4J_PASSWORD):
        raise RuntimeError(
            "NEO4J_USER and NEO4J_PASSWORD must be set in .env"
        )
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
    )


def build_app(
    graph: Neo4jGraph | None = None,
    llm: ChatOllama | None = None,
    *,
    with_validator: bool = False,
    max_validation_retries: int = 1,
) -> Any:
    """Compile the ReAct LangGraph application.

    Parameters
    ----------
    graph, llm
        Injectable for testing. Production callers can leave them None.
    with_validator
        If True, append a rule-based validator node after each agent turn.
        On violation, the validator routes back to the agent with up to
        ``max_validation_retries`` retries.
    max_validation_retries
        Hard cap on revision rounds. 1 is plenty for 7–9B models.
    """
    graph = graph or _connect_graph()
    llm = llm or ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    tools = build_tools(graph)
    llm_with_tools = llm.bind_tools(tools)

    # ---- agent node ------------------------------------------------------- #
    def call_model(state: AgentState) -> dict:
        messages = state.messages
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SYSTEM_PROMPT] + list(messages)
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # ---- validator node --------------------------------------------------- #
    def validate_node(state: AgentState) -> dict:
        # Most recent AIMessage is the candidate answer.
        ai_msgs = [m for m in state.messages if isinstance(m, AIMessage)]
        if not ai_msgs:
            return {"validation": ValidationResult(violations=[]),
                    "pending_retry": False}
        final = ai_msgs[-1]
        # If the agent is requesting more tools, defer validation —
        # tools_condition routes back to "tools" before this node runs.
        if final.tool_calls:
            return {"validation": ValidationResult(violations=[]),
                    "pending_retry": False}

        answer = final.content if isinstance(final.content, str) else str(final.content)

        # Reconstruct tool_calls from the message history.
        tool_calls = _extract_tool_calls(state.messages)
        result = validate_answer(answer, tool_calls, question=state.question)

        out: dict[str, Any] = {"validation": result, "pending_retry": False}
        if not result.ok and state.retry_count < max_validation_retries:
            # Use a HumanMessage rather than a SystemMessage so the chat
            # template sees a normal user-turn follow-up. Ollama's
            # tool-calling routing in llama3.1 is sensitive to multiple
            # system messages and can return empty completions when one
            # appears after the user message.
            out["messages"] = [HumanMessage(content=violations_message(result.violations))]
            out["retry_count"] = state.retry_count + 1
            out["pending_retry"] = True
        return out

    def _from_validate(state: AgentState) -> str:
        # The validator already enforced the budget when it set
        # pending_retry. We just route on that flag.
        return "agent" if state.pending_retry else END

    # ---- assemble the graph ---------------------------------------------- #
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "agent")

    if with_validator:
        workflow.add_node("validate", validate_node)
        # tools_condition routes:  agent -> tools (loop)  OR  agent -> validate
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tools", END: "validate"},
        )
        workflow.add_edge("tools", "agent")
        workflow.add_conditional_edges(
            "validate",
            _from_validate,
            {"agent": "agent", END: END},
        )
    else:
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")

    return workflow.compile()


def _extract_tool_calls(messages: list[AnyMessage]) -> list[dict]:
    """Walk the message history pairing AIMessage tool calls with the
    matching ToolMessage outputs."""
    pending: dict[str, dict] = {}
    out: list[dict] = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                tid = tc.get("id")
                if tid is None:
                    continue
                pending[tid] = {"name": tc["name"],
                                "args": tc.get("args", {}),
                                "output": None}
        elif isinstance(m, ToolMessage):
            entry = pending.pop(m.tool_call_id, None)
            if entry is not None:
                entry["output"] = m.content
                out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# Convenience entry point used by the evaluation runner                      #
# --------------------------------------------------------------------------- #
def ask(app: Any, question: str) -> dict:
    """Run one question through the agent and return a structured trace.

    Returns
    -------
    dict with:
        answer:     final assistant answer (str)
        tool_calls: list of {name, args, output}
        messages:   raw message history
        validation: list[str] of rule violations on the FINAL answer (post-retry)
        retry_count: how many self-correction passes were triggered
    """
    initial = AgentState(messages=[HumanMessage(content=question)],
                         question=question)
    final_state = app.invoke(initial)
    messages = final_state["messages"]
    tool_calls = _extract_tool_calls(messages)

    # The agent's last AIMessage is the answer. The validator may append a
    # follow-up HumanMessage between turns; iterating in reverse picks the
    # most recent AIMessage regardless.
    final_msg = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)),
        messages[-1],
    )
    answer = final_msg.content if hasattr(final_msg, "content") else str(final_msg)

    validation = final_state.get("validation")
    retry_count = final_state.get("retry_count", 0)

    return {
        "answer": answer,
        "tool_calls": tool_calls,
        "messages": messages,
        "validation": list(validation.violations) if validation else [],
        "retry_count": retry_count,
    }


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*",
                        help="Question to ask. Defaults to a multi-hop probe.")
    parser.add_argument("--validator", action="store_true",
                        help="Enable the rule-based validator node.")
    args = parser.parse_args()

    print("Agentic GraphRAG (ReAct) — Neo4j + Ollama")
    print(f"  Neo4j:      {NEO4J_URI}")
    print(f"  Ollama:     {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
    print(f"  Tools:      {len(TOOL_CATALOG)} ({', '.join(TOOL_CATALOG)})")
    print(f"  Validator:  {'on' if args.validator else 'off'}")
    print()

    question = (
        " ".join(args.question)
        if args.question
        else "Which species share a habitat with the Sumatran orangutan and "
             "are threatened by habitat loss?"
    )
    print(f"User: {question}\n")

    app = build_app(with_validator=args.validator)
    result = ask(app, question)

    print("─── tool calls ───")
    for tc in result["tool_calls"]:
        print(f"  • {tc['name']}({tc['args']}) -> {tc['output']}")

    if result["validation"]:
        print(f"\n─── validator (after {result['retry_count']} retry/retries) ───")
        for v in result["validation"]:
            print(f"  ! {v}")

    print("\n─── final answer ───")
    print(result["answer"])
