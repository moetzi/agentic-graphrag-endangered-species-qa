# Endangered Species Agentic GraphRAG QA

**Author:** Mutiara Noor Fauzia — NRP 5026221045

An agentic GraphRAG question-answering system over a Neo4j knowledge graph
of endangered species scraped from the World Wildlife Fund (WWF) website.
The agent uses a ReAct loop driven by a local Ollama LLM and 18
schema-aware parametric Cypher tools, and is evaluated on a generated
ground-truth set of 30 single-hop and multi-hop QA pairs.

The project is organised so each deliverable maps to a clearly named file:

| Deliverable | Path |
|---|---|
| Project overview, install, run, architecture | this `README.md` |
| Architecture diagram (rendered on GitHub) | [`docs/architecture.md`](docs/architecture.md) |
| Cypher + AI pipeline explanation | [`docs/cypher_and_pipeline.md`](docs/cypher_and_pipeline.md) |
| Evaluation methodology + ground truth | [`evaluation/README.md`](evaluation/README.md) |
| Execution screenshots | [`docs/screenshots/`](docs/screenshots/) |
| AI assistance documentation | [`docs/ai_usage.md`](docs/ai_usage.md) |

---

## 1. Architecture at a glance

### System overview

```mermaid
flowchart TB
    subgraph DATA["Data layer"]
        SRC[("data/species.json<br/>46 species scraped from WWF")]
    end

    subgraph INGEST["Ingestion"]
        ING["ingestion.py<br/>UPSERT + UNIQUE constraints<br/>+ bidirectional SHARES_HABITAT_WITH"]
    end

    subgraph STORE["Storage - Docker"]
        NEO[("Neo4j 5.25 + APOC<br/>Species, Habitat, Threat, ConservationAction")]
    end

    subgraph AGENT["Agent - LangGraph ReAct"]
        direction TB
        Q["User question"]
        LLM["Agent node<br/>ChatOllama llama3.1 8B<br/>+ 18 bound tools"]
        TOOLS["ToolNode<br/>parametric Cypher"]
        VAL["Validator node<br/>3 rules, max 1 retry"]:::optional
        Q --> LLM
        LLM <-->|tool calls / ToolMessage| TOOLS
        LLM --> VAL
        VAL -->|violations| LLM
    end

    subgraph EVAL["Evaluation"]
        GT[("evaluation/ground_truth.json<br/>18 single-hop + 12 multi-hop")]
        RUN["evaluation/runner.py"]
        ABL["evaluation/run_ablation.py"]
        MET["Metrics:<br/>recall@k, precision@k,<br/>faithfulness, hallucination,<br/>tool selection, latency, tokens"]
    end

    SRC --> ING --> NEO
    NEO -.Cypher rows.-> TOOLS
    GT --> RUN --> MET
    GT --> ABL --> MET
    AGENT -.answer + tool trace.-> RUN
    AGENT -.answer + tool trace.-> ABL

    classDef optional stroke-dasharray: 5 5
```

### LangGraph state machine

```mermaid
stateDiagram-v2
    [*] --> agent
    agent --> tools: tool_calls present
    tools --> agent: ToolMessage outputs
    agent --> validate: final answer
    validate --> [*]: ok or retry budget hit
    validate --> agent: violations and retries left
```

### Knowledge graph schema

```mermaid
flowchart LR
    S1["Species<br/>name, scientific_name,<br/>status, description,<br/>population, weight"]
    S2["Species"]
    H["Habitat<br/>name"]
    T["Threat<br/>name"]
    A["ConservationAction<br/>name"]

    S1 -->|LIVES_IN| H
    S1 -->|THREATENED_BY| T
    S1 -->|PROTECTED_BY| A
    S1 <-->|SHARES_HABITAT_WITH| S2
```

### Tool-call sequence (typical multi-hop request)

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant L as LLM
    participant T as ToolNode
    participant N as Neo4j
    participant V as Validator

    U->>L: question
    L->>T: get_neighbors_by_threat<br/>species_name=Sumatran orangutan,<br/>threat_keyword=habitat loss
    T->>N: MATCH ...SHARES_HABITAT_WITH...<br/>WHERE toLower contains keyword
    N-->>T: rows
    T-->>L: ToolMessage list of species
    L-->>V: final AIMessage
    V-->>L: violations if any
    V-->>U: validated answer
```

### Evaluation ablation matrix

```mermaid
flowchart LR
    subgraph Conditions
        B["Base ReAct<br/>+ 18 tools"]
        V["+ Validator"]
    end
    Conditions --> R["evaluation/run_ablation.py"]
    R --> M["ablation_summary.md<br/>side-by-side metric table"]
```

## 2. Components

| File / dir | Role |
|---|---|
| `data/species.json` | Source dataset — 46 endangered species scraped from WWF. |
| `docker-compose.yml` | Neo4j 5.25 community + APOC, healthcheck, mem-tuned. |
| `ingestion.py` | UPSERT species → Neo4j with constraints + bidirectional `SHARES_HABITAT_WITH`. |
| `graphrag_qa.py` | LangGraph ReAct app: `START → agent ↔ tools → [validate] → END`. |
| `evaluation/tool_catalog.py` | 18 parametric Cypher tools; **single source of truth** for the agent and the gold dataset. |
| `evaluation/build_ground_truth.py` | Generates `ground_truth.json` directly from `species.json`. |
| `evaluation/validator.py` | Three-rule grounded-answer validator (pure Python). |
| `evaluation/metrics.py` | Pure-function metric implementations (recall@k, faithfulness, etc.). |
| `evaluation/runner.py` | End-to-end evaluator with two modes: `--gold-tools`, `--with-agent`. |
| `evaluation/run_ablation.py` | Drives the two ablation conditions (base / validator) and emits a comparison table. |
| `check_connectivity.py` | Preflight: env, Neo4j, Ollama, schema, model presence, chat round-trip. |

---

## 3. Installation

### 3.1 Prerequisites

* Python 3.12 (or 3.11+)
* Docker Desktop (for Neo4j)
* [Ollama](https://ollama.com/download) for local LLM inference
* ~4 GB free disk for the LLM weights

### 3.2 Clone and set up the Python env

```cmd
git clone https://github.com/moetzi/agentic-graphrag-endangered-species-qa.git
cd agentic-graphrag-endangered-species-qa

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3.3 Configure environment

Copy the template, then fill in your own Neo4j password:

```cmd
copy .env.example .env
:: edit .env -> NEO4J_PASSWORD = "<your-password>"
```

`docker-compose.yml` reads `NEO4J_USER` and `NEO4J_PASSWORD` from the same
`.env`, so you only configure them in one place.

### 3.4 Pull the LLM

In a separate terminal, leave `ollama serve` running, then pull the model
(default is `llama3.1` — an 8B parameter model that fits 7–9B target):

```cmd
ollama serve
ollama pull llama3.1
```

### 3.5 Start Neo4j

```cmd
docker compose up -d
docker compose ps                    :: wait until "healthy"
```

### 3.6 Verify everything is ready

```cmd
python check_connectivity.py
```

You should see all-PASS (or PASS + a `WARN: graph empty` if you haven't
ingested yet). The script prints actionable error messages if something
isn't running.

---

## 4. Running the pipeline

```cmd
:: 1. Ingest the data into Neo4j
python ingestion.py

:: 2. Generate the ground-truth dataset (if not already in repo)
python -m evaluation.build_ground_truth

:: 3. Sanity check — replay gold tool calls against Neo4j (no LLM)
python -m evaluation.runner --gold-tools

:: 4. Real run — full ReAct agent
python -m evaluation.runner --with-agent

:: 5. (Optional) Ask one question interactively
python graphrag_qa.py "What threats does the Sumatran orangutan face?"
python graphrag_qa.py --validator "Which species share a habitat with the Sumatran orangutan?"

:: 6. (Optional) Run the two-way ablation study (base vs. validator)
python -m evaluation.run_ablation
```

Each runner writes timestamped artefacts under `evaluation/runs/<ts>-<mode>/`:
`results.jsonl`, `summary.json`, `summary.md`. The `runs/` folder is
git-ignored.

---

## 5. Cypher logic and AI pipeline (in brief)

The full discussion is in
[`docs/cypher_and_pipeline.md`](docs/cypher_and_pipeline.md). The short
version:

* The graph is a property graph with four labels (`Species`, `Habitat`,
  `Threat`, `ConservationAction`) and four relationship types (`LIVES_IN`,
  `THREATENED_BY`, `PROTECTED_BY`, `SHARES_HABITAT_WITH`). Every label has
  a `UNIQUE` constraint on `name`, so `MERGE` is idempotent and re-runnable.
* All retrieval is done through 18 **parametric** Cypher templates in
  [`evaluation/tool_catalog.py`](evaluation/tool_catalog.py). Each template
  uses query parameters (`$species_name`, `$threat_keyword`, …) — no string
  interpolation, no Cypher injection risk, easy to evaluate.
* The LangGraph ReAct loop in `graphrag_qa.py` exposes those 18 tools to
  the LLM. The LLM picks a tool per turn, observes the result, and either
  calls another tool or emits a final answer.
* One optional ablation component: a rule-based **validator** that
  checks whether every entity in the answer was actually returned by a
  tool, re-prompting the agent on violation (1-retry cap).

---

## 6. Screenshots 

### (a) Database connection 

![Neo4j connection check — all-PASS output](docs/screenshots/01_neo4j_connection.png)

---

### (b) Query result

![Neo4j graph — Species and Habitat subgraph](docs/screenshots/02_neo4j_graph.png)

![Neo4j graph — alternate view](docs/screenshots/02_neo4j_graph1.png)

![Neo4j graph — multi-hop edges](docs/screenshots/02_neo4j_graph3.png)

---

### (c) Evaluation / analysis output

![Evaluation summary — metric digest](docs/screenshots/03_evaluation_summary1.png)

**Ablation comparison** — 30 items, k = 10

| Condition | Answer Correctness | Faithfulness | Hallucination Rate | Recall@10 | Precision@10 | Tool Exact Seq | Tool Set Match | Validator Fired | Validator Retried | Latency (s) | Tokens Total | Wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.9532 | 0.7374 | 0.2626 | 0.8728 | 0.87 | 0.90 | 0.90 | 0.0 | 0.0 | 5.07 | 1434.77 | 157.87 |
| validator | 0.9082 | 0.7432 | 0.2568 | 0.8728 | 0.87 | 0.37 | 0.90 | 0.6 | 0.6 | 6.40 | 2491.80 | 192.79 |

#### Analysis — Validator Over-Correction in a Small Local Model

The ablation reveals a classic **over-correction failure mode** that emerges when a rule-based validator is paired with a small (8 B parameter) local model.

**The validator fires too aggressively.** With `validator_fired = 0.60` and `validator_retried = 0.60`, the three-rule grounding check triggered a retry on 18 out of 30 questions — far more than expected for a model whose base answer correctness is already 0.9532. This signals that the validator's strictness threshold is not well-calibrated for `llama3.1:8b`'s output style.

**Retrying hurts answer quality.** Answer correctness falls from **0.9532 → 0.9082** (−4.7 pp) after adding the validator. The re-prompt forces the model to regenerate its answer under an additional grounding constraint it cannot reliably satisfy at 8 B scale, producing outputs that score lower than the original uncorrected answer.

**Tool order collapses while tool coverage is preserved.** `tool_exact_sequence` drops sharply from **0.90 → 0.37** (−58.9 pp), yet `tool_set_match` stays at **0.90**. This means the model calls the same set of tools during the retry but reorders them unpredictably — the validator changes *when* tools are invoked without changing *which* tools are chosen. Larger models tend to maintain consistent ordering under re-prompting; 8 B models do not.

**Faithfulness and hallucination improve only marginally.** Faithfulness rises by just **+0.006** (0.7374 → 0.7432) and hallucination rate drops by **−0.006** (0.2626 → 0.2568). The grounding benefit is real but negligible — far too small to justify the cost.

**The overhead is substantial.** Tokens consumed increase by **+73.7 %** (1 434 → 2 491 per question) and per-question latency grows by **+26 %** (5.07 s → 6.40 s). For a production deployment of a small local model these costs are prohibitive relative to the ~0.6 pp reduction in hallucination rate achieved.

**Takeaway.** The validator adds value when paired with a capable model that can reliably comply with re-prompting instructions. For `llama3.1:8b`, the retry loop costs significantly more than it gains: answer quality degrades, token spend nearly doubles, and the only measurable benefit — a marginal reduction in hallucination — does not offset the losses. A lighter intervention (e.g., a lower-threshold validator or a single-shot grounding prefix) would be more appropriate for this model size.

---

### (d) LLM / RAG / Cypher agent demo

![Agent demo — tool calls and final answer](docs/screenshots/04_agent_demo.png)

---

## 7. AI usage disclosure

This project was built collaboratively with the **Claude Opus 4.7 (1M
context preview)** assistant inside Kiro IDE. The full prompt log,
model identifier, and a per-file summary of which parts were AI-drafted
vs hand-edited live in [`docs/ai_usage.md`](docs/ai_usage.md), as required
by the assignment.

The headline: AI was used to draft scaffolding, generator code, the metric
implementations, and most of the Cypher templates. Every file was reviewed
and modified by hand. The dataset, design choices (e.g., parametric tools
over text-to-Cypher; rule-based validator instead of an LLM critic), and
all evaluation methodology decisions were made by the author.

---

## 8. Repository conventions

* `.env` is git-ignored. Use `.env.example` as the template.
* Neo4j data and logs (`neo4j_data/`, `neo4j_logs/`, `neo4j_conf/`) are
  git-ignored — they're local Docker volumes.
* Evaluation runs (`evaluation/runs/`) are git-ignored. Only the
  generated `ground_truth.json` is committed.
* `docker-compose.yml` reads credentials from `.env` so there is one
  source of truth for both the DB and Python.

---

## 9. Tech stack

| Layer | Library | Version |
|---|---|---|
| Graph DB | Neo4j Community | 5.25 |
| Graph driver | `neo4j` (Python) | 6.1 |
| LLM | Ollama (`llama3.1` 8B) | latest |
| Agent framework | LangGraph | 1.1 |
| Tool binding | LangChain (`langchain-ollama`, `langchain-community`) | 1.x / 0.4.x |
| Schema validation | Pydantic | 2.12 |
| Container | Docker Compose | latest |

See [`requirements.txt`](requirements.txt) for the exact pins.
