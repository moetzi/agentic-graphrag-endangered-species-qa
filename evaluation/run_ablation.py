"""
Run the two ablation conditions sequentially and emit a comparison table.

Conditions:
    base       - ReAct over parametric Cypher tools (current default)
    validator  - + rule-based grounded-answer validator (1 retry)

Usage:
    python -m evaluation.run_ablation
    python -m evaluation.run_ablation --limit 5
    python -m evaluation.run_ablation --conditions base validator

Outputs:
    evaluation/runs/<ts>-ablation/
    ├── <condition>/results.jsonl   per-condition per-item details
    ├── <condition>/summary.json    per-condition aggregated metrics
    └── ablation_summary.md         side-by-side comparison
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from evaluation.runner import (
    GROUND_TRUTH_PATH, RUNS_DIR,
    _run_agent, _aggregate, _render_markdown,
)

CONDITIONS = {
    "base":      dict(with_validator=False),
    "validator": dict(with_validator=True),
}


def _print_table(rows: list[dict], cols: list[str]) -> str:
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows))
              for c in cols}
    head = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    lines = [head, sep]
    for r in rows:
        lines.append(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="*",
                   choices=list(CONDITIONS), default=list(CONDITIONS),
                   help="Subset of conditions to run (default: both).")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--price-in", type=float, default=0.0)
    p.add_argument("--price-out", type=float, default=0.0)
    args = p.parse_args()

    gt = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    items = gt["items"]
    if args.ids:
        wanted = set(args.ids)
        items = [it for it in items if it["id"] in wanted]
    if args.limit:
        items = items[: args.limit]

    base_dir = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-ablation"
    base_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    for cond in args.conditions:
        flags = CONDITIONS[cond]
        sub = base_dir / cond
        sub.mkdir(exist_ok=True)
        print(f"\n=== condition: {cond} ===")
        print(f"   flags: {flags}")
        t0 = time.perf_counter()
        results = _run_agent(items, args.k, args.price_in, args.price_out, **flags)
        wall = time.perf_counter() - t0

        with (sub / "results.jsonl").open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary = _aggregate(results, args.k)
        summary["condition"] = cond
        summary["wall_seconds"] = round(wall, 2)
        (sub / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (sub / "summary.md").write_text(
            _render_markdown(summary, cond, sub), encoding="utf-8")
        summaries[cond] = summary

    # Comparison table
    cols = ["condition", "answer_correctness", "faithfulness",
            "hallucination_rate", f"recall_at_{args.k}",
            f"precision_at_{args.k}", "tool_exact_sequence", "tool_set_match",
            "validator_fired", "validator_retried",
            "latency_seconds", "tokens_total", "wall_seconds"]
    rows = []
    for cond, s in summaries.items():
        row = {"condition": cond, "wall_seconds": s.get("wall_seconds", "-")}
        for c in cols[1:-1]:
            row[c] = s["overall"].get(c, "-")
        rows.append(row)

    table = _print_table(rows, cols)

    print("\n=== ablation comparison ===")
    print(table)

    md_lines = [
        f"# Ablation comparison — {base_dir.name}",
        "",
        f"- items: {len(items)}",
        f"- k: {args.k}",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in rows:
        md_lines.append("| " + " | ".join(str(r.get(c, "-")) for c in cols) + " |")
    (base_dir / "ablation_summary.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\nWrote {base_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
