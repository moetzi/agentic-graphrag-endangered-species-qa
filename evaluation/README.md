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

## Running the evaluator

```cmd
:: 1. Sanity check — replay the gold tool calls against Neo4j (no LLM).
::    Confirms the live graph is consistent with ground_truth.json.
python -m evaluation.runner --gold-tools

:: 2. Real run — the LangGraph agent answers each question end-to-end.
python -m evaluation.runner --with-agent

:: 3. Iterate quickly on a subset.
python -m evaluation.runner --with-agent --ids sh-001 mh-005
python -m evaluation.runner --with-agent --limit 5

:: 4. Add token pricing for the cost column (defaults to 0 for local Ollama).
python -m evaluation.runner --with-agent --price-in 0.50 --price-out 1.50
```

Each run writes:

```
evaluation/runs/<timestamp>-<mode>/
├── results.jsonl   # per-item trace + metrics
├── summary.json    # aggregated overall + per hop / category / difficulty
└── summary.md      # human-readable report
```

`evaluation/runs/` is git-ignored.

### Metric semantics

| Metric                | Source / formula                                                                    |
|-----------------------|--------------------------------------------------------------------------------------|
| `answer_correctness`  | string answers: gold ⊆ answer (case-folded). set answers: F1 over gold mentions.     |
| `entity_recall`       | gold_entities mentioned in the answer / |gold_entities|.                            |
| `faithfulness`        | mentioned entities supported by retrieved tool outputs (substring-aware).           |
| `hallucination_rate`  | 1 − faithfulness.                                                                    |
| `recall@k`            | retrieved (top-k) ∩ gold_answer / |gold_answer|. Capped when |gold_answer| > k.     |
| `precision@k`         | retrieved (top-k) ∩ gold_answer / k.                                                |
| `context_precision`   | retrieved items overlapping the gold answer set (substring-aware).                  |
| `tool_exact_sequence` | predicted tool name sequence == any expected / alternative sequence.                |
| `tool_set_match`      | set of called tool names == set of expected tool names.                             |
| `tool_args_match`     | among matched-sequence runs, fraction of tool calls with arg-equal payloads.        |
| `latency_seconds`     | wall-clock of `app.invoke(...)` for that question.                                   |
| `tokens_input/output/total` | summed `usage_metadata` across every `AIMessage` in the trace.                |
| `cost_usd`            | tokens_input × price_in / 1k + tokens_output × price_out / 1k.                       |

## Ablation study

Two optional components can be toggled around the ReAct core:

| Component   | What it does                                                                              | Cost                          |
|-------------|--------------------------------------------------------------------------------------------|-------------------------------|
| Planner     | Classifies the question's hop & category (regex-first, LLM fallback) and injects a hint.  | 0 tokens (regex) / ~80 tokens (LLM fallback). |
| Validator   | Runs three rules over the agent's answer; on violation, asks the agent to revise (1 retry max). | 0 tokens (rules) / one extra LLM call only when a violation fires. |

Run conditions individually:

```cmd
:: Base (just ReAct)
python -m evaluation.runner --with-agent --ablation base

:: + planner
python -m evaluation.runner --with-agent --ablation planner

:: + validator
python -m evaluation.runner --with-agent --ablation validator

:: + both
python -m evaluation.runner --with-agent --ablation both
```

…or run all four in one shot and emit a comparison table:

```cmd
python -m evaluation.run_ablation
python -m evaluation.run_ablation --limit 5
python -m evaluation.run_ablation --conditions base both
```

The ablation runner writes a per-condition subdirectory plus an
`ablation_summary.md` with a side-by-side metric comparison.

### New metrics introduced by the ablation components

| Metric                | Source                                                                  |
|-----------------------|--------------------------------------------------------------------------|
| `plan_hop_match`      | 1 if the planner's `hop` matches the gold item's `hop`, else 0.        |
| `plan_category_match` | 1 if the planner's `category` matches the gold item's `category`, else 0. |
| `validator_fired`     | 1 if the validator detected at least one violation on the first pass. |
| `validator_retried`   | 1 if the validator triggered the self-correction retry.                |
