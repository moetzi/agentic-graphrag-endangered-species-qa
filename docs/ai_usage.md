# AI usage disclosure

This file documents how generative AI was used while building this
project, as required by the assignment ("Penggunaan AI untuk generate
code diperbolehkan, namun harus didokumentasikan").

## 1. Tool and model

| Field | Value |
|---|---|
| Assistant | Kiro IDE built-in agent |
| Model | **Claude Opus 4.7** (1M context preview) |
| Mode | Autopilot — assistant proposes file edits and shell commands; the user approves and reviews each turn. |
| Date range | 2026-06 |

## 2. High-level prompts used

These are paraphrased from the chat transcript — the actual conversation
was iterative and conversational. Each bullet is a distinct request that
produced or modified files.

1. *"Build an agentic GraphRAG QA system for endangered species using
   data scraped from WWF (`data/species.json`), Neo4j in Docker, Ollama
   for LLM inference, and a ReAct loop with schema-aware parametric
   Cypher tools. First, create a ground truth for single-hop and
   multi-hop QA so we can evaluate the system later."*
2. *"Continue building ground truth, limit to max 30 QA pairs."*
3. *"Apply fix to `ingestion.py` for what's required to evaluate using
   the ground truth."*
4. *"Fix `graphrag_qa.py` so the program works smoothly to test the
   ground truth and ensure tools listed match `tool_catalog.py`."*
5. *"Try `pip install` for dependencies first since I don't install
   anything, or just create `requirements.txt`."*
6. *"Add a `.gitignore` and `.env.example`."*
7. *"Fix `docker-compose.yml`."*
8. *"Add the GitHub remote and push the current changes."*
9. *"Build the evaluation runner."*
10. *"Create `check_connectivity.py`."*
11. *"Since I apply a ReAct agent here, is it valid to call this project
    'agentic GraphRAG'? — discussion only."*
12. *"Implement my recommended scenarios for the ablation study"*
    (planner + rule-based validator).
13. *"My deliverables should be …"* (this README + `docs/` folder).

The full conversation transcript is preserved in the IDE's session log.

## 3. What the AI generated vs what was hand-written

| File | AI-drafted | Hand-modified |
|---|---|---|
| `data/species.json` | ❌ | ✅ Author scraped + curated. |
| `data_pipeline.py` (legacy, deleted) | ✅ | ✅ Removed by the author. |
| `docker-compose.yml` | ✅ initial draft | ✅ author flagged the empty `NEO4J_AUTH`, stale mount, and Swarm-only `deploy:`; AI rewrote with fixes verified by `docker compose config`. |
| `ingestion.py` | ✅ (AI rewrote bug-fixed version) | ✅ Author specified bidirectional `SHARES_HABITAT_WITH` requirement. |
| `graphrag_qa.py` | ✅ (LangGraph wiring, planner/validator integration) | ✅ Author chose ReAct + parametric tools, decided against text-to-Cypher. |
| `evaluation/tool_catalog.py` | ✅ | ✅ Author defined the schema and tool taxonomy first; AI translated to Cypher templates. |
| `evaluation/build_ground_truth.py` | ✅ | ✅ Author specified 18 single-hop + 12 multi-hop split, chose categories. |
| `evaluation/ground_truth.json` | generator output | reviewed by author |
| `evaluation/metrics.py` | ✅ | ✅ Author flagged metric semantics (e.g., recall@k uses `gold_answer`, not `gold_entities`); AI corrected. |
| `evaluation/runner.py` | ✅ | ✅ Author requested two run modes (`--gold-tools`, `--with-agent`). |
| `evaluation/planner.py` | ✅ | ✅ Author chose regex-first approach over LLM-only. |
| `evaluation/validator.py` | ✅ | ✅ Author chose rule-based over LLM-critic. |
| `evaluation/run_ablation.py` | ✅ | ✅ Author specified the four ablation conditions. |
| `check_connectivity.py` | ✅ | ✅ Author requested per-component checks with actionable error messages. |
| `requirements.txt` | ✅ | ✅ Author asked for pinned versions. |
| `.gitignore`, `.env.example` | ✅ | reviewed |
| `README.md`, `docs/*.md` | ✅ | reviewed; this file written based on author's chat transcript. |

## 4. Design decisions made by the author (not AI)

These were *human* decisions, taken after AI raised options:

* **Use parametric Cypher tools instead of text-to-Cypher generation.**
  Justification: a 7–9B local model (llama3.1) makes text-to-Cypher
  expensive and accuracy-poor; parametric tools give deterministic
  execution traces for the evaluation metrics.
* **Use a regex-first planner.** Tested: regex matches 30/30 of the
  ground-truth questions; LLM fallback exists only for robustness.
* **Use a rule-based validator instead of an LLM critic.** Pure-Python
  rules are free and don't require another model call; cap retries at
  1 to bound latency.
* **Keep `SHARES_HABITAT_WITH` bidirectional but skip unresolved
  neighbours** (e.g., "Krill") rather than auto-creating placeholder
  Species nodes.
* **Standardise on `neo4j` as the bootstrap admin user** rather than a
  custom username — avoids needing to provision a second user inside
  the DB.
* **Generator-driven ground truth, not hand-written JSON.** When the
  source dataset changes, regenerating the ground truth keeps gold
  answers consistent.

## 5. Manual verification steps performed

* `python check_connectivity.py` — component-level smoke test.
* `python -m evaluation.build_ground_truth --check` — drift guard.
* Offline metric tests with fabricated traces (perfect + adversarial)
  to confirm the metric math.
* Offline ablation smoke test with a stub LLM to confirm the LangGraph
  topology under all four condition flags.
* `docker compose config` to validate compose interpolation and verify
  no real credentials are echoed beyond the local `.env`.
* `git check-ignore` before the first commit to confirm `.env` was not
  staged.

## 6. Known limitations

* The 18 tools cover the full schema but cannot answer questions outside
  the schema (e.g., dietary relationships). This is by design — the
  scope is bounded to the WWF source dataset.
* The validator's proper-noun extractor is a simple capitalisation
  heuristic. False negatives are acceptable (validator is conservative);
  false positives would manifest as unnecessary retries. Empirically the
  rules fire on real hallucinations and don't fire on grounded answers.
* Token cost is reported assuming a configurable per-1k rate. For local
  Ollama the rate is 0; the field exists so the same harness can be
  pointed at a hosted API later.
