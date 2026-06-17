"""
Pure-function metric computations for the agentic GraphRAG evaluator.

Everything here is deterministic and dependency-free so unit tests stay simple
and the runner can be reused with any agent that returns the same trace shape.

A "trace" is the dict produced by `graphrag_qa.ask`:
    {
        "answer":     str,
        "tool_calls": [{"name": str, "args": dict, "output": Any}, ...],
        "messages":   list[AnyMessage],
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


# --------------------------------------------------------------------------- #
# Normalisation                                                               #
# --------------------------------------------------------------------------- #
def _norm(s: Any) -> str:
    """Case-fold + strip + collapse whitespace for string comparison."""
    return " ".join(str(s).split()).lower()


def _norm_set(items: Iterable[Any]) -> set[str]:
    return {_norm(x) for x in items if x is not None and str(x).strip()}


# --------------------------------------------------------------------------- #
# Tool-output handling                                                        #
# --------------------------------------------------------------------------- #
def _flatten_tool_output(output: Any) -> list[str]:
    """Tool outputs come back as list[str], a single str, or '[]'-ish.

    Convert to a flat list of strings for set operations.
    """
    if output is None:
        return []
    if isinstance(output, list):
        return [str(x) for x in output if x is not None]
    if isinstance(output, str):
        # ToolNode serialises list[str] back to a string like "['Sumatran...']".
        # Try a permissive parse; otherwise treat the whole thing as one item.
        s = output.strip()
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1]
            parts = [p.strip().strip("'\"") for p in inner.split(",")]
            return [p for p in parts if p]
        return [s] if s else []
    return [str(output)]


def retrieved_entities(tool_calls: list[dict]) -> list[str]:
    """All distinct entities returned by tool calls, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for call in tool_calls:
        for item in _flatten_tool_output(call.get("output")):
            key = _norm(item)
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out


# --------------------------------------------------------------------------- #
# Set-style retrieval metrics (recall@k, precision@k, F1)                     #
# --------------------------------------------------------------------------- #
def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 1.0
    g = _norm_set(gold)
    r = _norm_set(retrieved[:k])
    return len(r & g) / len(g)


def precision_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    g = _norm_set(gold)
    top = retrieved[:k]
    if not top:
        return 0.0 if g else 1.0
    r = _norm_set(top)
    return len(r & g) / len(r)


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# Answer correctness                                                          #
# --------------------------------------------------------------------------- #
def answer_correctness(answer: str, gold: Any, answer_type: str) -> float:
    """`string` answers: 1.0 if gold is contained in the answer, else 0.0.
    `set` answers: F1 over the gold entities mentioned in the answer.
    """
    answer_norm = _norm(answer)

    if answer_type == "string":
        return 1.0 if _norm(gold) in answer_norm else 0.0

    # set
    gold_set = _norm_set(gold)
    if not gold_set:
        return 1.0
    mentioned = {g for g in gold_set if g in answer_norm}
    precision = len(mentioned) / len(gold_set) if mentioned else 0.0
    recall = len(mentioned) / len(gold_set)
    # We don't know the false-positive set easily here, so use recall as the
    # set-membership-precision floor (every match is in gold, so precision=1
    # over the matched subset). This is a slight optimism but gives a usable
    # F1 per question.
    precision = 1.0 if mentioned else 0.0
    return f1(precision, recall)


def entity_recall(answer: str, gold_entities: list[str]) -> float:
    g = _norm_set(gold_entities)
    if not g:
        return 1.0
    a = _norm(answer)
    return sum(1 for e in g if e in a) / len(g)


def entity_precision(answer: str, gold_entities: list[str]) -> float:
    """Fraction of *gold* entities mentioned in the answer that are correct.

    Trivially 1.0 by construction (any gold entity in the answer is correct);
    kept for symmetry with `entity_recall`. Use `faithfulness` for the
    grounded version (penalises ungrounded mentions).
    """
    g = _norm_set(gold_entities)
    if not g:
        return 1.0
    a = _norm(answer)
    mentioned = sum(1 for e in g if e in a)
    return 1.0 if mentioned else 0.0


# --------------------------------------------------------------------------- #
# Faithfulness / hallucination                                                #
# --------------------------------------------------------------------------- #
def faithfulness(answer: str,
                 gold_entities: list[str],
                 retrieved: list[str]) -> float:
    """Among entities the answer mentions (drawn from gold ∪ retrieved),
    what fraction are supported by retrieval?

    "Supported" = entity is a substring of (or equal to) any retrieved item,
    using the same case-folded matching as "mentioned". Symmetric matching
    avoids penalising the agent for keywords like "Africa" that appear inside
    longer retrieved entities like "African forest elephant".

    Returns 1.0 when the answer doesn't mention any tracked entity.
    """
    a = _norm(answer)
    grounded = _norm_set(gold_entities) | _norm_set(retrieved)
    mentioned = {e for e in grounded if e in a}
    if not mentioned:
        return 1.0
    retrieved_norm = _norm_set(retrieved)
    retrieved_blob = " || ".join(retrieved_norm)  # for substring checks
    supported = {e for e in mentioned
                 if e in retrieved_norm or e in retrieved_blob}
    return len(supported) / len(mentioned)


def hallucination_rate(answer: str,
                       gold_entities: list[str],
                       retrieved: list[str]) -> float:
    return 1.0 - faithfulness(answer, gold_entities, retrieved)


def context_precision(retrieved: list[str], gold_entities: list[str]) -> float:
    """Fraction of retrieved items that overlap (substring or equal) with
    the gold entity set. Substring matching keeps short keyword golds (like
    "Africa") from penalising rich retrieved items ("African forest elephant").
    """
    if not retrieved:
        return 1.0 if not gold_entities else 0.0
    r = _norm_set(retrieved)
    g = _norm_set(gold_entities)
    if not g:
        return 1.0
    gold_blob = " || ".join(g)
    hits = sum(1 for x in r if x in g or any(x in gx or gx in x for gx in g))
    return hits / len(r)


# --------------------------------------------------------------------------- #
# Tool-selection metrics                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class ToolSelectionScore:
    exact_sequence: float = 0.0   # 1 if predicted name-sequence == any expected
    set_match: float = 0.0        # 1 if set(names) == set(expected_names)
    args_match: float = 0.0       # avg fraction of matched tools whose args fold-equal
    matched_alternative: int = -1 # which alternative matched (-1 = none)


def _tool_call_signature(call: dict) -> tuple[str, dict]:
    return call["name"], {k: _norm(v) for k, v in (call.get("args") or {}).items()}


def score_tool_selection(predicted: list[dict],
                         expected: list[dict],
                         alternatives: list[list[dict]] | None = None
                         ) -> ToolSelectionScore:
    pred_sigs = [_tool_call_signature(c) for c in predicted]
    pred_names = [n for n, _ in pred_sigs]

    candidates: list[list[dict]] = [expected] + (alternatives or [])
    score = ToolSelectionScore()

    # exact sequence (names only) and args match against any candidate
    for idx, cand in enumerate(candidates):
        cand_sigs = [_tool_call_signature(c) for c in cand]
        cand_names = [n for n, _ in cand_sigs]
        if pred_names == cand_names:
            score.exact_sequence = 1.0
            score.matched_alternative = idx
            # args match per-tool
            arg_hits = sum(
                1 for (_, pa), (_, ca) in zip(pred_sigs, cand_sigs) if pa == ca
            )
            score.args_match = arg_hits / len(cand_sigs) if cand_sigs else 1.0
            break

    # set match: any candidate's name-set equals predicted name-set
    pred_set = set(pred_names)
    if any(set(n for n, _ in (_tool_call_signature(c) for c in cand)) == pred_set
           for cand in candidates):
        score.set_match = 1.0

    return score


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #
def mean(values: Iterable[float]) -> float:
    vs = list(values)
    return sum(vs) / len(vs) if vs else 0.0
