# Ground Truth — Endangered Species Agentic GraphRAG

This folder contains the gold dataset used to evaluate the agentic GraphRAG QA
system end-to-end.

## Contents

```
evaluation/
├── tool_catalog.py          # Reference catalogue of parametric Cypher tools
├── build_ground_truth.py    # Generates ground_truth.json from data/species.json
└── ground_truth.json        # 30 QA pairs (18 single-hop + 12 multi-hop)
```

## Why generate, not hand-write?

Gold answers are derived directly from `data/species.json`. If you re-scrape or
edit the source, simply rerun the generator and the gold answers stay correct:

```bash
python -m evaluation.build_ground_truth          # regenerate
python -m evaluation.build_ground_truth --check  # CI-friendly drift check
```

## Schema of every QA item

| Field               | Type            | Purpose                                                                                             |
|---------------------|-----------------|-----------------------------------------------------------------------------------------------------|
| `id`                | str             | Stable identifier (`sh-###` single-hop, `mh-###` multi-hop).                                        |
| `hop`               | `single`/`multi`| Reasoning depth.                                                                                    |
| `category`          | str             | Semantic bucket (e.g., `common_threats`).                                                           |
| `difficulty`        | str             | `easy` / `medium` / `hard`.                                                                         |
| `question`          | str             | Natural-language question fed to the agent.                                                         |
| `answer_type`       | `string`/`set`  | How to compare predicted vs gold.                                                                   |
| `gold_answer`       | str / list[str] | Reference answer, sorted for set-type answers.                                                      |
| `gold_entities`     | list[str]       | Entities expected to appear in the answer (used for entity-level recall).                           |
| `gold_contexts`     | list[str]       | Atomic facts the agent must rely on (used for context precision/faithfulness).                      |
| `expected_tools`    | list[ToolCall]  | Tools the agent is expected to call, with `name`, `args`, and the parametric `cypher` template.     |
| `alternative_tools` | list[list[ToolCall]] _(optional)_ | Acceptable alternative tool sequences (e.g., decomposing into 2 single-hop calls).|

## How each metric maps to the dataset

| Metric                | Compute from                                                                |
|-----------------------|-----------------------------------------------------------------------------|
| Recall@k              | `gold_entities` ∩ predicted entities in retrieved contexts (top-k).         |
| Precision@k           | predicted entities ∩ `gold_entities` over k retrieved contexts.             |
| Answer relevance      | embedding similarity between agent answer and `gold_answer`.                |
| Faithfulness          | fraction of agent claims supported by `gold_contexts`.                      |
| Context precision     | fraction of retrieved contexts that overlap `gold_contexts`.                |
| Hallucination rate    | 1 − faithfulness, or share of entities not in `gold_entities`.              |
| Tool selection        | sequence match against `expected_tools` / `alternative_tools`.              |
| Latency               | wall-clock per question (instrument the agent loop).                        |
| Token cost            | sum of input + output tokens reported by the LLM client.                    |

## Distribution

- 18 single-hop: 4 status, 3 scientific name, 3 habitats, 3 threats, 2 conservation
  actions, 1 find-by-threat, 1 find-by-habitat, 1 find-by-status.
- 12 multi-hop: 4 neighbours, 3 common threats, 1 neighbours-by-threat, 1
  threat+habitat, 1 threat+status, 2 actions-for-threat.

## ⚠️ One ingestion gap to fix

`graphrag_qa.py` (and many gold multi-hop items) depend on a
`SHARES_HABITAT_WITH` relation between `Species` nodes. `ingestion.py` does not
yet create it. Add the following to `ingestion.py` so the multi-hop ground truth
is reachable end-to-end:

```python
# Treat shares_habitat_with as bidirectional.
FOREACH (n IN $shares_habitat_with |
    MERGE (other:Species {name: n})
    MERGE (s)-[:SHARES_HABITAT_WITH]->(other)
    MERGE (other)-[:SHARES_HABITAT_WITH]->(s))
```

…and pass `shares_habitat_with=data.get('shares_habitat_with', [])` to
`session.run`. Note that some `shares_habitat_with` entries name species that
are not in `species.json` (e.g., "Krill"). The generator already filters those
out of the gold answers, so they won't pollute recall/precision.
