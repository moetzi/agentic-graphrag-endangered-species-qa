# Cypher logic and AI pipeline

This document explains how data flows from raw JSON into a knowledge
graph, and how the agent uses parametric Cypher tools to answer
questions.

## 1. Graph schema

```
(:Species {name, scientific_name, status, description,
           population, weight, height, length})

(:Habitat {name})
(:Threat {name})
(:ConservationAction {name})

(:Species)-[:LIVES_IN]->(:Habitat)
(:Species)-[:THREATENED_BY]->(:Threat)
(:Species)-[:PROTECTED_BY]->(:ConservationAction)
(:Species)-[:SHARES_HABITAT_WITH]->(:Species)   -- bidirectional
```

Why these labels and not, say, `Region` or `Diet`? The dataset (scraped
from WWF) consistently exposes only these four entity classes, and they
cover every QA pattern in the ground truth. Adding speculative labels
would create empty subgraphs and dilute retrieval metrics.

UNIQUE constraints are applied before any `MERGE` so the ingestion is
idempotent and fast on re-runs:

```cypher
CREATE CONSTRAINT species_name IF NOT EXISTS
FOR (s:Species) REQUIRE s.name IS UNIQUE
-- (same for Habitat, Threat, ConservationAction)
```

## 2. Ingestion logic (`ingestion.py`)

Two passes:

1. **Pass 1 — UPSERT each species + its scalar fan-outs** in one Cypher
   statement using `FOREACH`:

   ```cypher
   MERGE (s:Species {name: $name})
   SET s.scientific_name = $scientific_name,
       s.status          = $status,
       s.description     = $description,
       s.population      = $population, ...

   FOREACH (h IN $habitats |
       MERGE (hab:Habitat {name: h})
       MERGE (s)-[:LIVES_IN]->(hab))

   FOREACH (t IN $threats |
       MERGE (thr:Threat {name: t})
       MERGE (s)-[:THREATENED_BY]->(thr))

   FOREACH (a IN $actions |
       MERGE (act:ConservationAction {name: a})
       MERGE (s)-[:PROTECTED_BY]->(act))
   ```

2. **Pass 2 — bidirectional `SHARES_HABITAT_WITH`** between species that
   already exist as nodes. We deliberately don't auto-create Species
   nodes for unresolved entries (e.g., "Krill", "Tiger") so the graph
   stays aligned with the source dataset and the ground truth's
   resolvable-neighbours filter.

   ```cypher
   MATCH (a:Species {name: $species})
   UNWIND $neighbors AS neighbor_name
   MATCH (b:Species {name: neighbor_name})
   MERGE (a)-[:SHARES_HABITAT_WITH]->(b)
   MERGE (b)-[:SHARES_HABITAT_WITH]->(a)
   ```

   This keeps the relation symmetric so multi-hop tools work regardless
   of which endpoint is queried.

## 3. The 18 parametric Cypher tools

All retrieval is funnelled through 18 tools defined in
[`evaluation/tool_catalog.py`](../evaluation/tool_catalog.py). They split
into three families:

### Single-hop, anchored on a species

```python
get_species_status        # MATCH (s:Species {name: $species_name}) RETURN s.status
get_species_scientific_name
get_species_description
get_habitats_of_species
get_threats_of_species
get_conservation_actions_of_species
```

### Single-hop, anchored on a property

```python
find_species_by_status       # toLower CONTAINS toLower
find_species_by_threat
find_species_by_habitat
find_species_by_conservation_action
```

### Multi-hop joins

```python
get_neighbors_of_species             # via SHARES_HABITAT_WITH
get_neighbors_by_threat              # neighbours filtered by a threat keyword
get_threats_of_neighbors
get_common_threats                   # intersection of two species' threats
find_species_by_threat_and_habitat
find_species_by_threat_and_status
get_conservation_actions_for_threat
find_habitats_for_threat
```

Each tool is a fixed parameterised template — the LLM never writes raw
Cypher. This buys three things:

1. **No injection risk** — `$species_name` is a real Cypher parameter.
2. **Deterministic execution traces** — `tool_exact_sequence` and
   `tool_args_match` metrics become meaningful only when the LLM picks
   from a closed vocabulary of tools.
3. **Cheap on a 7–9B model** — text-to-Cypher with small open models is
   accuracy-poor and iteration-heavy. Parametric templates short-circuit
   that loop. (See `docs/ai_usage.md` for the discussion.)

## 4. AI pipeline

```
question
   │
   ▼
(agent: ChatOllama(llama3.1) bound to the 18 tools)
   │   ReAct: reason → choose tool → observe → repeat → final answer.
   │
   ▼
(tools: ToolNode running parametric Cypher against Neo4j)
   │
   ▼
(agent: synthesises the final NL answer from tool outputs)
   │
   ▼
(validator — optional, pure Python)
   │   3 rules:
   │   (a) "answer makes factual claim but no tool was called"
   │   (b) "proper-noun phrase in answer not in retrieved tool outputs"
   │   (c) "number in answer not present in any retrieved value"
   │   On violation: append a HumanMessage describing the violations
   │   and route back to the agent (max 1 retry).
   │
   ▼
final answer + tool-call trace
```

> **Earlier iteration:** an optional planner node prepended a regex- or
> LLM-derived hop+category hint. On `llama3.1:8B` with bound tools, that
> second `SystemMessage` after the user turn caused the model to skip
> tool-calling entirely and emit a one-token reply. The component was
> removed; the validator stayed because it works.

## 5. Why ReAct, not a fixed pipeline

A fixed RAG pipeline would have to either (a) always run all relevant
tools and let the LLM filter — wasteful on multi-hop questions — or (b)
hard-code a tool per question category, which collapses the system to
template matching with extra steps.

The ReAct loop lets the model decompose a multi-hop question into the
minimum number of tool calls, observe an empty result and either reframe
or stop. This is why questions like "common threats of A and B" can be
answered either via the composite `get_common_threats(species_a,
species_b)` tool or via two `get_threats_of_species(...)` calls plus an
intersection — both alternatives are recorded in the ground truth and
both are scored as correct tool selections.

## 6. Evaluation hook-in

The runner re-uses `evaluation.tool_catalog.TOOL_CATALOG` to:

* expose the same 18 tools to the agent (no drift)
* re-execute the ground truth's `expected_tools` directly against Neo4j
  in `--gold-tools` mode, providing a sanity check that decouples
  graph correctness from agent quality.

`tool_catalog.py` is therefore the single source of truth for both the
agent's runtime tools and the evaluator's expected tools — there is no
way for them to drift apart.
