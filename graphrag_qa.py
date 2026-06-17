"""
Agentic GraphRAG over Neo4j with a ReAct loop driven by Ollama.

Tools are derived from `evaluation.tool_catalog.TOOL_CATALOG` so the agent's
runtime tools cannot drift from the ground truth's `expected_tools`.

Run a single question (CLI):
    python graphrag_qa.py "What threats does the Sumatran orangutan face?"

Programmatic entry point used by the evaluation runner:
    from graphrag_qa import build_app, ask
    app = build_app()
    result = ask(app, "...question...")
    # result -> {"answer": str, "tool_calls": [...], "messages": [...]}
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
from langgraph.graph import StateGraph, START      # noqa: E402
from langgraph.graph.message import add_messages   # noqa: E402
from langgraph.prebuilt import ToolNode, tools_condition  # noqa: E402

from evaluation.tool_catalog import TOOL_CATALOG   # noqa: E402

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
) -> Any:
    """Compile the ReAct LangGraph application.

    Both `graph` and `llm` are injectable so the evaluation runner can swap in
    a mock graph or a different model without re-wiring the topology.
    """
    graph = graph or _connect_graph()
    llm = llm or ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    tools = build_tools(graph)
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: AgentState) -> dict:
        messages = state.messages
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SYSTEM_PROMPT] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    return workflow.compile()


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
    """
    initial = AgentState(messages=[HumanMessage(content=question)])
    final_state = app.invoke(initial)
    messages = final_state["messages"]

    tool_calls: list[dict] = []
    pending: dict[str, dict] = {}
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                pending[tc["id"]] = {
                    "name": tc["name"],
                    "args": tc.get("args", {}),
                    "output": None,
                }
        elif isinstance(m, ToolMessage):
            entry = pending.get(m.tool_call_id)
            if entry is not None:
                entry["output"] = m.content
                tool_calls.append(entry)

    final_msg = messages[-1]
    answer = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
    return {"answer": answer, "tool_calls": tool_calls, "messages": messages}


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Agentic GraphRAG (ReAct) — Neo4j + Ollama")
    print(f"  Neo4j:  {NEO4J_URI}")
    print(f"  Ollama: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
    print(f"  Tools:  {len(TOOL_CATALOG)} ({', '.join(TOOL_CATALOG)})")
    print()

    question = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Which species share a habitat with the Sumatran orangutan and "
             "are threatened by habitat loss?"
    )
    print(f"User: {question}\n")

    app = build_app()
    result = ask(app, question)

    print("─── tool calls ───")
    for tc in result["tool_calls"]:
        print(f"  • {tc['name']}({tc['args']}) -> {tc['output']}")

    print("\n─── final answer ───")
    print(result["answer"])
