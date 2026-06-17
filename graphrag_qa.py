"""
Agentic GraphRAG over Neo4j with a ReAct loop driven by Ollama.

Tools are derived from `evaluation.tool_catalog.TOOL_CATALOG` so the agent's
runtime tools cannot drift from the ground truth's `expected_tools`.

Two optional ablation components can be toggled at build time:

    build_app(with_planner=True)    # prepend a hop/category hint
    build_app(with_validator=True)  # rule-based grounded-answer check
                                    # with one self-correction retry

Run a single question (CLI):
    python graphrag_qa.py "What threats does the Sumatran orangutan face?"
    python graphrag_qa.py --planner --validator "..."

Programmatic entry point used by the evaluation runner:
    from graphrag_qa import build_app, ask
    app = build_app(with_planner=True, with_validator=True)
    result = ask(app, "...question...")
    # result -> {"answer": str, "tool_calls": [...], "messages": [...],
    #            "plan": Plan|None, "validation": [...]}
"""
from __future__ import annotations

import os
import sys
from typing import Annotated, Any, Callable

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

from evaluation.planner import Plan, plan as classify_plan  # noqa: E402
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
    # Planner / validator metadata (None when their nodes are disabled).
    plan: Plan | None = None
    validation: ValidationResult | None = None
    retry_count: int = 0
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
    with_planner: bool = False,
    with_validator: bool = False,
    max_validation_retries: int = 1,
) -> Any:
    """Compile the ReAct LangGraph application.

    Parameters
    ----------
    graph, llm
        Injectable for testing. Production callers can leave them None.
    with_planner
        If True, prepend a planner node that classifies the question's
        hop/category and injects a SystemMessage hint before the agent runs.
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

    # ---- planner node ----------------------------------------------------- #
    def planner_node(state: AgentState) -> dict:
        question = state.question or ""
        # First HumanMessage backs out the question if it wasn't passed in.
        if not question:
            for m in state.messages:
                if isinstance(m, HumanMessage):
                    question = m.content if isinstance(m.content, str) else str(m.content)
                    break
        p = classify_plan(question, llm_fallback=llm)
        hint_msg = SystemMessage(
            content=f"Planner hint ({p.backend}, {p.hop}-hop, "
                    f"category={p.category}): {p.hint}"
        )
        return {"plan": p, "messages": [hint_msg], "question": question}

    # ---- agent node ------------------------------------------------------- #
    def call_model(state: AgentState) -> dict:
        messages = state.messages
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SYSTEM_PROMPT] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # ---- validator node --------------------------------------------------- #
    def validate_node(state: AgentState) -> dict:
        # Most recent AIMessage is the candidate answer.
        ai_msgs = [m for m in state.messages if isinstance(m, AIMessage)]
        if not ai_msgs:
            return {"validation": ValidationResult(violations=[])}
        final = ai_msgs[-1]
        # If the agent is requesting more tools, defer validation —
        # tools_condition routes back to "tools" before this node runs.
        if final.tool_calls:
            return {"validation": ValidationResult(violations=[])}

        answer = final.content if isinstance(final.content, str) else str(final.content)

        # Reconstruct tool_calls from the message history.
        tool_calls = _extract_tool_calls(state.messages)
        result = validate_answer(answer, tool_calls, question=state.question)

        out: dict[str, Any] = {"validation": result}
        if not result.ok and state.retry_count < max_validation_retries:
            out["messages"] = [SystemMessage(content=violations_message(result.violations))]
            out["retry_count"] = state.retry_count + 1
        return out

    def _from_validate(state: AgentState) -> str:
        v = state.validation
        # Stop if no violations OR we've already used our retry budget.
        if v is None or v.ok:
            return END
        if state.retry_count >= max_validation_retries:
            return END
        return "agent"

    # ---- assemble the graph ---------------------------------------------- #
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    if with_planner:
        workflow.add_node("planner", planner_node)
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "agent")
    else:
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
                pending[tc["id"]] = {"name": tc["name"],
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
        plan:       Plan instance or None (only present if planner was on)
        validation: list[str] of rule violations on the FINAL answer (post-retry)
        retry_count: how many self-correction passes were triggered
    """
    initial = AgentState(messages=[HumanMessage(content=question)],
                         question=question)
    final_state = app.invoke(initial)
    messages = final_state["messages"]
    tool_calls = _extract_tool_calls(messages)

    # The agent's last AIMessage is the answer. The validator may append a
    # SystemMessage after it (a nag for the next iteration); skip those.
    final_msg = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)),
        messages[-1],
    )
    answer = final_msg.content if hasattr(final_msg, "content") else str(final_msg)

    plan_obj = final_state.get("plan")
    validation = final_state.get("validation")
    retry_count = final_state.get("retry_count", 0)

    return {
        "answer": answer,
        "tool_calls": tool_calls,
        "messages": messages,
        "plan": plan_obj,
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
    parser.add_argument("--planner", action="store_true",
                        help="Enable the hop/category planner node.")
    parser.add_argument("--validator", action="store_true",
                        help="Enable the rule-based validator node.")
    args = parser.parse_args()

    print("Agentic GraphRAG (ReAct) — Neo4j + Ollama")
    print(f"  Neo4j:      {NEO4J_URI}")
    print(f"  Ollama:     {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
    print(f"  Tools:      {len(TOOL_CATALOG)} ({', '.join(TOOL_CATALOG)})")
    print(f"  Planner:    {'on' if args.planner else 'off'}")
    print(f"  Validator:  {'on' if args.validator else 'off'}")
    print()

    question = (
        " ".join(args.question)
        if args.question
        else "Which species share a habitat with the Sumatran orangutan and "
             "are threatened by habitat loss?"
    )
    print(f"User: {question}\n")

    app = build_app(with_planner=args.planner, with_validator=args.validator)
    result = ask(app, question)

    if result["plan"]:
        p = result["plan"]
        print(f"─── plan ─── ({p.backend})")
        print(f"  hop={p.hop} category={p.category}")
        print(f"  hint: {p.hint}")
        print()

    print("─── tool calls ───")
    for tc in result["tool_calls"]:
        print(f"  • {tc['name']}({tc['args']}) -> {tc['output']}")

    if result["validation"]:
        print(f"\n─── validator (after {result['retry_count']} retry/retries) ───")
        for v in result["validation"]:
            print(f"  ! {v}")

    print("\n─── final answer ───")
    print(result["answer"])
