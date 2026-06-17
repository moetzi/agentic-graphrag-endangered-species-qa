# Execution screenshots

The assignment asks for four execution screenshots. Capture each one as
described and save it under this folder using the suggested filename, so
the grader can match them to the rubric.

## (a) Database connection — `01_neo4j_connection.png`

**Easiest option:** terminal output of `python check_connectivity.py`
showing all-PASS rows.

**Alternative:** Neo4j Browser at `http://localhost:7474` after logging
in with your `.env` credentials, with the schema panel visible in the
left sidebar.

```cmd
python check_connectivity.py
```

## (b) Query / graph builder result — `02_neo4j_graph.png`

Open Neo4j Browser at `http://localhost:7474`, run a query that returns
a visual subgraph, then screenshot the result. Suggested queries:

```cypher
:: full schema
CALL db.schema.visualization()

:: a small slice of the data
MATCH (s:Species)-[:LIVES_IN]->(h:Habitat)
RETURN s, h LIMIT 30

:: multi-hop edges (bidirectional sharing)
MATCH (a:Species)-[:SHARES_HABITAT_WITH]-(b:Species)
RETURN a, b
```

## (c) Analysis / ML output — `03_evaluation_summary.png`

Run the evaluator and screenshot the terminal at the "Overall:" digest:

```cmd
python -m evaluation.runner --gold-tools
:: or the full agent run:
python -m evaluation.runner --with-agent
```

If you also run the ablation:

```cmd
python -m evaluation.run_ablation
```

…include the side-by-side comparison table near the bottom of its
console output as `04_ablation_table.png` (optional bonus).

## (d) LLM / RAG / Cypher demo — `04_agent_demo.png`

A single end-to-end CLI run that shows the tool calls and the final
answer (the validator output if any rules trigger):

```cmd
python graphrag_qa.py --validator "Which species share a habitat with the Sumatran orangutan and are threatened by habitat loss?"
```

Capture the full console output (tool calls list → optional validator
block → final answer).

## Optional architecture diagram

The repo's architecture diagrams are written in Mermaid in
[`docs/architecture.md`](../architecture.md) and render directly on
GitHub, so a separate PNG isn't required for the rubric. If your
submission needs a static image (e.g., a printed report), export a
PNG using <https://mermaid.live/> and save it as `architecture.png`
in this folder.

## Filename convention

```
docs/screenshots/
├── 01_neo4j_connection.png
├── 02_neo4j_graph.png
├── 03_evaluation_summary.png
├── 04_agent_demo.png
└── architecture.png            # optional Excalidraw export
```

Sticking to these names lets the grader see at a glance which deliverable
each capture satisfies.
