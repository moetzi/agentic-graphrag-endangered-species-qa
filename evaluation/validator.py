"""
Rule-based grounded-answer validator for the agentic GraphRAG agent.

Runs *after* the agent emits its final answer. Pure Python — no LLM call.
If any rule fires, the validator returns a list of violations the
LangGraph wrapper turns into a SystemMessage telling the agent to revise.

Bound the retry loop at 1 in the LangGraph wrapper so a stubborn agent
can't loop indefinitely.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from evaluation.metrics import _flatten_tool_output, _norm, _norm_set


# Stop-words / phrases that mean "I have nothing" — a graceful refusal is OK
# and shouldn't trip the "asserts facts but never used a tool" rule.
_REFUSAL_MARKERS = (
    "no data found", "no data available", "i don't know",
    "no information", "not found", "cannot determine",
    "could not find", "unable to find",
)


@dataclass
class ValidationResult:
    violations: list[str]

    @property
    def ok(self) -> bool:
        return not self.violations


def _retrieved_set(tool_calls: list[dict]) -> set[str]:
    out: set[str] = set()
    for c in tool_calls:
        for item in _flatten_tool_output(c.get("output")):
            out.add(_norm(item))
    return out


def _capitalised_phrases(text: str) -> set[str]:
    """Cheap proper-noun extractor for grounded-claim auditing.

    Captures runs of Capitalised words ("Sumatran orangutan", "African forest
    elephant"). Imperfect but sufficient: false negatives are fine because
    the validator is conservative by design.
    """
    pattern = re.compile(r"\b[A-Z][a-zA-Z]+(?:[\s-][A-Z][a-zA-Z]+){0,3}\b")
    out: set[str] = set()
    for m in pattern.findall(text or ""):
        norm = _norm(m)
        # filter out common throwaways at sentence starts
        if norm in {"the", "a", "an", "i", "this", "these", "those", "however"}:
            continue
        if len(norm) < 3:
            continue
        out.add(norm)
    return out


# --------------------------------------------------------------------------- #
# Individual rules                                                            #
# --------------------------------------------------------------------------- #
def rule_no_tool_for_factual_claim(answer: str,
                                   tool_calls: list[dict]) -> str | None:
    """If the answer asserts something concrete but no tool was ever called,
    that is by construction a hallucination."""
    if tool_calls:
        return None
    a = _norm(answer)
    if any(m in a for m in _REFUSAL_MARKERS):
        return None
    if len(a) < 8:  # extremely short answers, treat as refusal
        return None
    return ("Answer asserts facts but no tool was called. "
            "Use one of the provided tools to ground the answer, or say "
            "explicitly that no data was found.")


def rule_proper_nouns_grounded(answer: str,
                               tool_calls: list[dict],
                               extra_anchors: set[str]) -> str | None:
    """Every proper-noun phrase in the answer should appear (possibly as a
    substring) in retrieved tool outputs OR in the user's question itself
    (so anchors like "Sumatran orangutan" don't trip the rule)."""
    retrieved = _retrieved_set(tool_calls)
    if not retrieved:
        return None  # rule_no_tool_for_factual_claim handles this case
    blob = " || ".join(retrieved) + " || " + " || ".join(extra_anchors)

    nouns = _capitalised_phrases(answer)
    ungrounded = []
    for noun in nouns:
        if noun in retrieved or noun in blob:
            continue
        ungrounded.append(noun)
    if not ungrounded:
        return None
    return ("Answer contains entities not returned by any tool: "
            f"{sorted(ungrounded)}. Drop them or call a tool to verify.")


def rule_numbers_must_be_retrieved(answer: str,
                                   tool_calls: list[dict]) -> str | None:
    """Numeric claims must come from a tool output. Population counts,
    specific dates, etc. None of the current 18 tools return numerics so
    this rule is a strict 'remove the number' guard.
    """
    nums = re.findall(r"\b\d{2,}\b", answer or "")
    if not nums:
        return None
    retrieved_blob = " ".join(_retrieved_set(tool_calls))
    leaked = [n for n in nums if n not in retrieved_blob]
    if not leaked:
        return None
    return (f"Answer contains numbers {leaked} that no tool returned. "
            "Remove the figures or rely only on tool output.")


# --------------------------------------------------------------------------- #
# Aggregator                                                                  #
# --------------------------------------------------------------------------- #
def validate(answer: str,
             tool_calls: list[dict],
             question: str = "") -> ValidationResult:
    """Run every rule and collect violations."""
    anchors = _norm_set(_capitalised_phrases(question))

    rules = [
        rule_no_tool_for_factual_claim(answer, tool_calls),
        rule_proper_nouns_grounded(answer, tool_calls, anchors),
        rule_numbers_must_be_retrieved(answer, tool_calls),
    ]
    return ValidationResult(violations=[r for r in rules if r])


def violations_message(violations: list[str]) -> str:
    """Format a SystemMessage that nudges the agent to revise."""
    bullets = "\n".join(f"- {v}" for v in violations)
    return (
        "The previous answer failed automated grounding checks:\n"
        f"{bullets}\n\n"
        "Revise the answer using ONLY facts returned by the provided tools. "
        "If no tool result supports a claim, remove it. If you have nothing "
        "to ground the answer, reply: 'No data found.'"
    )
