"""
Evaluation runner for the agentic GraphRAG QA system.

Two execution modes:

    --with-agent   Run the real LangGraph agent (Ollama + Neo4j) for each
                   question. Slow but produces the actual end-to-end metrics.

    --gold-tools   Skip the agent. Execute the gold `expected_tools` against
                   Neo4j and treat the result as the agent's answer. Useful
                   for verifying the ground truth is reachable in the live
                   graph before any agent tuning.

Outputs (under evaluation/runs/<timestamp>/):
    results.jsonl   per-question trace + metrics
    summary.json    aggregated metrics (overall + per hop / category)
    summary.md      human-readable report

Usage examples:
    python -m evaluation.runner --gold-tools
    python -m evaluation.runner --with-agent
    python -m evaluation.runner --with-agent --limit 5 --ids sh-001 mh-005
    python -m evaluation.runner --with-agent --price-in 0 --price-out 0
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from evaluation import metrics

ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_PATH = ROOT / "evaluation" / "ground_truth.json"
RUNS_DIR = ROOT / "evaluation" / "runs"


# --------------------------------------------------------------------------- #
# Per-question evaluation                                                     #
# --------------------------------------------------------------------------- #
def _token_usage(messages: list[Any]) -> dict[str, int]:
    """Sum usage_metadata across every AIMessage in the trace."""
    inp = out = total = 0
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "usage_metadata", None):
            u = m.usage_metadata or {}
            inp   += int(u.get("input_tokens", 0) or 0)
            out   += int(u.get("output_tokens", 0) or 0)
            total += int(u.get("total_tokens", 0) or 0)
    if not total:
        total = inp + out
    return {"input": inp, "output": out, "total": total}


def evaluate_item(item: dict,
                  trace: dict,
                  *,
                  k: int,
                  price_in_per_1k: float,
                  price_out_per_1k: float,
                  latency_seconds: float) -> dict:
    """Compute the full metric record for one ground-truth item + trace."""
    answer        = trace.get("answer", "") or ""
    tool_calls    = trace.get("tool_calls", []) or []
    messages      = trace.get("messages", []) or []
    retrieved     = metrics.retrieved_entities(tool_calls)
    gold_answer   = item["gold_answer"]
    gold_entities = item["gold_entities"]
    answer_type   = item["answer_type"]

    # Retrieval metrics measure "did we retrieve the answer-side entities".
    # Use gold_answer (a list for set-typed, [value] for string-typed) so we
    # don't penalise retrieval for missing query-anchor keywords.
    gold_answer_list = (gold_answer if isinstance(gold_answer, list)
                        else [gold_answer])

    tokens = _token_usage(messages)
    cost = (tokens["input"]  / 1000.0) * price_in_per_1k \
         + (tokens["output"] / 1000.0) * price_out_per_1k

    tool_score = metrics.score_tool_selection(
        predicted=tool_calls,
        expected=item["expected_tools"],
        alternatives=item.get("alternative_tools"),
    )

    # Planner / validator metrics (default to absent flags when disabled).
    plan_obj = trace.get("plan")
    plan_hop_match = plan_cat_match = None
    plan_backend = None
    if plan_obj is not None:
        plan_hop_match = 1.0 if plan_obj.hop == item["hop"] else 0.0
        plan_cat_match = 1.0 if plan_obj.category == item["category"] else 0.0
        plan_backend = plan_obj.backend

    validation_violations = trace.get("validation") or []
    retry_count = trace.get("retry_count", 0) or 0
    validator_fired = 1.0 if (validation_violations or retry_count > 0) else 0.0
    validator_retried = 1.0 if retry_count > 0 else 0.0

    record_metrics: dict[str, float] = {
        "answer_correctness":  metrics.answer_correctness(answer, gold_answer, answer_type),
        "entity_recall":       metrics.entity_recall(answer, gold_entities),
        "faithfulness":        metrics.faithfulness(answer, gold_entities, retrieved),
        "hallucination_rate":  metrics.hallucination_rate(answer, gold_entities, retrieved),
        f"recall_at_{k}":      metrics.recall_at_k(retrieved, gold_answer_list, k),
        f"precision_at_{k}":   metrics.precision_at_k(retrieved, gold_answer_list, k),
        "context_precision":   metrics.context_precision(retrieved, gold_answer_list),
        "tool_exact_sequence": tool_score.exact_sequence,
        "tool_set_match":      tool_score.set_match,
        "tool_args_match":     tool_score.args_match,
        "latency_seconds":     latency_seconds,
        "tokens_input":        tokens["input"],
        "tokens_output":       tokens["output"],
        "tokens_total":        tokens["total"],
        "cost_usd":            cost,
    }
    if plan_obj is not None:
        record_metrics["plan_hop_match"] = plan_hop_match
        record_metrics["plan_category_match"] = plan_cat_match
    record_metrics["validator_fired"]   = validator_fired
    record_metrics["validator_retried"] = validator_retried

    return {
        "id":        item["id"],
        "hop":       item["hop"],
        "category":  item["category"],
        "difficulty": item["difficulty"],
        "question":  item["question"],
        "gold_answer": gold_answer,
        "answer":    answer,
        "tool_calls": [
            {"name": c["name"], "args": c.get("args", {}), "output": c.get("output")}
            for c in tool_calls
        ],
        "expected_tool_names": [t["name"] for t in item["expected_tools"]],
        "predicted_tool_names": [c["name"] for c in tool_calls],
        "plan": (
            {"hop": plan_obj.hop, "category": plan_obj.category,
             "backend": plan_backend, "hint": plan_obj.hint}
            if plan_obj is not None else None
        ),
        "validation_violations": validation_violations,
        "retry_count": retry_count,
        "metrics": record_metrics,
    }


# --------------------------------------------------------------------------- #
# Run modes                                                                   #
# --------------------------------------------------------------------------- #
def _run_agent(items: list[dict], k: int,
               price_in_per_1k: float, price_out_per_1k: float,
               *, with_planner: bool = False,
               with_validator: bool = False) -> list[dict]:
    """Mode: drive the real LangGraph agent."""
    from graphrag_qa import build_app, ask  # imported lazily to avoid Neo4j on dry runs
    app = build_app(with_planner=with_planner, with_validator=with_validator)
    out: list[dict] = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item['id']}: {item['question']}")
        t0 = time.perf_counter()
        try:
            trace = ask(app, item["question"])
        except Exception as exc:  # surface the failure but keep going
            print(f"   ! agent error: {exc}")
            trace = {"answer": f"<agent error: {exc}>", "tool_calls": [],
                     "messages": [], "plan": None, "validation": [],
                     "retry_count": 0}
        latency = time.perf_counter() - t0
        out.append(evaluate_item(
            item, trace,
            k=k,
            price_in_per_1k=price_in_per_1k,
            price_out_per_1k=price_out_per_1k,
            latency_seconds=latency,
        ))
    return out


def _run_gold_tools(items: list[dict], k: int) -> list[dict]:
    """Mode: replay the ground-truth tool calls directly against Neo4j.

    Skips the LLM entirely so we can prove the gold dataset is consistent
    with the live graph before any agent tuning.
    """
    from graphrag_qa import _connect_graph
    graph = _connect_graph()
    out: list[dict] = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item['id']}: replaying gold tools")
        t0 = time.perf_counter()
        tool_calls: list[dict] = []
        for tc in item["expected_tools"]:
            rows = graph.query(tc["cypher"], params=tc["args"])
            if rows:
                first_key = next(iter(rows[0].keys()))
                output = [r[first_key] for r in rows if r.get(first_key) is not None]
            else:
                output = []
            tool_calls.append({"name": tc["name"], "args": tc["args"], "output": output})

        # Synthesise an answer from the joined tool outputs so the
        # answer-level metrics still have something to score.
        joined = sorted({str(x) for c in tool_calls for x in (c["output"] or [])})
        if item["answer_type"] == "string":
            answer = ", ".join(joined) if joined else "No data found."
        else:
            answer = "; ".join(joined) if joined else "No data found."
        latency = time.perf_counter() - t0

        out.append(evaluate_item(
            item,
            trace={"answer": answer, "tool_calls": tool_calls, "messages": []},
            k=k,
            price_in_per_1k=0.0,
            price_out_per_1k=0.0,
            latency_seconds=latency,
        ))
    return out


# --------------------------------------------------------------------------- #
# Aggregation & reporting                                                     #
# --------------------------------------------------------------------------- #
_NUMERIC_FIELDS = (
    "answer_correctness", "entity_recall", "faithfulness", "hallucination_rate",
    "context_precision", "tool_exact_sequence", "tool_set_match", "tool_args_match",
    "latency_seconds", "tokens_input", "tokens_output", "tokens_total", "cost_usd",
)


def _aggregate(results: list[dict], k: int) -> dict:
    # Take the union of metric keys across all rows so optional fields
    # (plan_*) are included only when present.
    fields = sorted({key for r in results for key in r["metrics"]})

    def _bucket(rows: list[dict]) -> dict:
        out: dict[str, float] = {}
        for f in fields:
            vals = [r["metrics"][f] for r in rows if f in r["metrics"]]
            if vals:
                out[f] = round(metrics.mean(vals), 4)
        return out

    overall = _bucket(results)

    by_hop: dict[str, dict] = {}
    by_cat: dict[str, dict] = {}
    by_diff: dict[str, dict] = {}
    grouped_hop: dict[str, list[dict]] = defaultdict(list)
    grouped_cat: dict[str, list[dict]] = defaultdict(list)
    grouped_diff: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        grouped_hop[r["hop"]].append(r)
        grouped_cat[r["category"]].append(r)
        grouped_diff[r["difficulty"]].append(r)
    for h, rows in grouped_hop.items():
        by_hop[h] = {"count": len(rows), **_bucket(rows)}
    for c, rows in grouped_cat.items():
        by_cat[c] = {"count": len(rows), **_bucket(rows)}
    for d, rows in grouped_diff.items():
        by_diff[d] = {"count": len(rows), **_bucket(rows)}

    return {
        "count": len(results),
        "overall": overall,
        "by_hop": by_hop,
        "by_category": by_cat,
        "by_difficulty": by_diff,
        "k": k,
    }


def _render_markdown(summary: dict, mode: str, run_dir: Path) -> str:
    k = summary["k"]
    lines: list[str] = []
    lines.append(f"# Evaluation report — {run_dir.name}")
    lines.append("")
    lines.append(f"- mode: `{mode}`")
    lines.append(f"- items: **{summary['count']}**")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k_, v in summary["overall"].items():
        lines.append(f"| {k_} | {v} |")
    lines.append("")
    lines.append("## By hop")
    lines.append("")
    cols = ["count", "answer_correctness", "faithfulness",
            f"recall_at_{k}", f"precision_at_{k}",
            "tool_exact_sequence", "tool_set_match", "latency_seconds"]
    lines.append("| hop | " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for hop, row in summary["by_hop"].items():
        lines.append(f"| {hop} | " + " | ".join(str(row.get(c, "-")) for c in cols) + " |")
    lines.append("")
    lines.append("## By category")
    lines.append("")
    lines.append("| category | count | answer_correctness | faithfulness | tool_exact_sequence |")
    lines.append("|---|---|---|---|---|")
    for cat, row in sorted(summary["by_category"].items()):
        lines.append(f"| {cat} | {row['count']} | "
                     f"{row['answer_correctness']} | {row['faithfulness']} | "
                     f"{row['tool_exact_sequence']} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--with-agent", action="store_true",
                      help="Run the real LangGraph agent (Ollama + Neo4j).")
    mode.add_argument("--gold-tools", action="store_true",
                      help="Skip the agent; execute gold expected_tools "
                           "against Neo4j as a sanity check.")
    p.add_argument("--k", type=int, default=10,
                   help="k for recall@k / precision@k (default 10).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only run the first N items (for fast iteration).")
    p.add_argument("--ids", nargs="*", default=None,
                   help="Only run these specific item ids (e.g. sh-001 mh-005).")
    p.add_argument("--price-in", type=float, default=0.0,
                   help="USD per 1k input tokens (default 0 for local Ollama).")
    p.add_argument("--price-out", type=float, default=0.0,
                   help="USD per 1k output tokens (default 0 for local Ollama).")
    p.add_argument("--planner", action="store_true",
                   help="Enable the planner node (only used with --with-agent).")
    p.add_argument("--validator", action="store_true",
                   help="Enable the validator node (only used with --with-agent).")
    p.add_argument("--ablation", choices=["base", "planner", "validator", "both"],
                   default=None,
                   help="Convenience preset for ablation runs. Overrides "
                        "--planner/--validator if set.")
    p.add_argument("--out", type=Path, default=None,
                   help="Override the run output directory.")
    args = p.parse_args()

    gt = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    items: list[dict] = gt["items"]

    if args.ids:
        wanted = set(args.ids)
        items = [it for it in items if it["id"] in wanted]
        missing = wanted - {it["id"] for it in items}
        if missing:
            print(f"error: unknown ids: {sorted(missing)}", file=sys.stderr)
            return 2

    if args.limit is not None:
        items = items[: args.limit]

    if not items:
        print("error: no items selected", file=sys.stderr)
        return 2

    if args.ablation:
        args.planner   = args.ablation in ("planner", "both")
        args.validator = args.ablation in ("validator", "both")

    mode_label = "with-agent" if args.with_agent else "gold-tools"
    if args.with_agent:
        mode_label += f"-{args.ablation or ('p' if args.planner else '') + ('v' if args.validator else '') or 'base'}"
    run_dir = args.out or (RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{mode_label}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode:      {mode_label}")
    print(f"Items:     {len(items)} / {gt['counts']['total']}")
    print(f"Run dir:   {run_dir.relative_to(ROOT)}")
    if args.with_agent:
        print(f"Planner:   {'on' if args.planner else 'off'}")
        print(f"Validator: {'on' if args.validator else 'off'}")
    print()

    t0 = time.perf_counter()
    if args.with_agent:
        results = _run_agent(items, args.k, args.price_in, args.price_out,
                             with_planner=args.planner,
                             with_validator=args.validator)
    else:
        results = _run_gold_tools(items, args.k)
    wall = time.perf_counter() - t0

    # ---- write per-item results --------------------------------------------
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- aggregate ----------------------------------------------------------
    summary = _aggregate(results, args.k)
    summary["mode"] = mode_label
    summary["wall_seconds"] = round(wall, 2)
    if results:
        latencies = [r["metrics"]["latency_seconds"] for r in results]
        summary["latency_p50"] = round(statistics.median(latencies), 4)
        summary["latency_p95"] = round(
            sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)], 4
        )

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(
        _render_markdown(summary, mode_label, run_dir), encoding="utf-8"
    )

    # ---- console digest -----------------------------------------------------
    print()
    print(f"Done in {wall:.1f}s. Wrote {run_dir.relative_to(ROOT)}/")
    print()
    print("Overall:")
    for kf, v in summary["overall"].items():
        print(f"  {kf:24s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
