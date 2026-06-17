# Architecture diagram

GitHub renders Mermaid blocks natively, so this page stays as a single
source of truth — no separate PNG export needed. Each diagram below is
self-contained and can be edited in place.

## 1. System overview

End-to-end data flow from the WWF source dataset through ingestion, the
agent, and the evaluator.

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

## 2. LangGraph state machine

Exact compiled topology of `graphrag_qa.build_app(with_validator=True)`.

```mermaid
stateDiagram-v2
    [*] --> agent
    agent --> tools: tool_calls present
    tools --> agent: ToolMessage outputs
    agent --> validate: final answer
    validate --> [*]: ok or retry budget hit
    validate --> agent: violations and retries left
```

When the validator is disabled the `agent --> validate` edge collapses
to `agent --> [*]`.

## 3. Knowledge graph schema

The four labels and four relationship types Neo4j stores after
ingestion. Each label has a `UNIQUE` constraint on `name`. Two
`Species` nodes are shown so the bidirectional `SHARES_HABITAT_WITH`
edge is visible.

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

## 4. Tool-call sequence (typical multi-hop request)

A walkthrough of the question *"Which species share a habitat with the
Sumatran orangutan and are threatened by habitat loss?"*

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

## 5. Evaluation ablation matrix

The two conditions exercised by `evaluation/run_ablation.py`.

```mermaid
flowchart LR
    subgraph Conditions
        B["Base ReAct<br/>+ 18 tools"]
        V["+ Validator"]
    end
    Conditions --> R["evaluation/run_ablation.py"]
    R --> M["ablation_summary.md<br/>side-by-side metric table"]
```

---

## Optional: export to a static image

If you also want a static PNG (e.g., for a slide deck or printed
report), you have three options:

1. **GitHub** renders Mermaid inline already, so no export is needed
   for the rubric.
2. **Mermaid Live** — paste any of the blocks above into
   <https://mermaid.live/> and `Actions → PNG`.
3. **Mermaid CLI** — `npm install -g @mermaid-js/mermaid-cli` then
   `mmdc -i diagram.mmd -o diagram.png`.

Drop the resulting PNG into `docs/screenshots/architecture.png` if you
prefer a binary deliverable.
