"""
Hop / category planner for the agentic GraphRAG agent.

Two backends:

* ``regex`` (default) — deterministic, free, well-suited to a small open
  model where an extra LLM call is expensive. Patterns are matched in
  order of specificity (most-specific first).
* ``llm`` (optional fallback) — used only when the regex finds no match.
  One classification call returning a small JSON blob.

Planner outputs a :class:`Plan` that the LangGraph planner node converts
into a single ``SystemMessage`` hint prepended to the agent's context.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

Hop = Literal["single", "multi"]

# Recognised categories — must stay in lock-step with build_ground_truth.py.
CATEGORIES: tuple[str, ...] = (
    # single-hop
    "species_status",
    "scientific_name",
    "habitats_of_species",
    "threats_of_species",
    "conservation_actions_of_species",
    "find_by_threat",
    "find_by_habitat",
    "find_by_status",
    # multi-hop
    "neighbours_of_species",
    "common_threats",
    "neighbours_by_threat",
    "threat_and_habitat",
    "threat_and_status",
    "actions_for_threat",
    # fallback
    "unknown",
)


@dataclass
class Plan:
    hop: Hop
    category: str            # one of CATEGORIES
    hint: str                # short instruction the agent will see
    backend: str = "regex"   # "regex" | "llm" | "default"


# --------------------------------------------------------------------------- #
# Regex patterns                                                              #
# Order matters: more-specific patterns must come first so a multi-hop        #
# question like "Which species share a habitat with X and are threatened by   #
# Y?" doesn't get caught by the simpler "share a habitat with X" pattern.     #
# --------------------------------------------------------------------------- #
_PATTERNS: list[tuple[re.Pattern, tuple[Hop, str, str]]] = [
    # ---- multi-hop ----
    (
        re.compile(r"\bshare\s+a\s+habitat\s+with\b.+\band\s+(?:are\s+)?(?:also\s+)?threatened\s+by\b", re.I),
        ("multi", "neighbours_by_threat",
         "Multi-hop: find species sharing a habitat AND filter by threat. "
         "Use get_neighbors_by_threat with both arguments."),
    ),
    (
        re.compile(r"\bwhich\s+species\s+share\s+a\s+habitat\s+with\b", re.I),
        ("multi", "neighbours_of_species",
         "Multi-hop: list habitat neighbours of one species. "
         "Use get_neighbors_of_species."),
    ),
    (
        re.compile(r"\bthreats?\s+(?:are\s+)?faced\s+by\s+both\b", re.I),
        ("multi", "common_threats",
         "Multi-hop: intersect the threats of two species. "
         "Use get_common_threats with species_a and species_b."),
    ),
    (
        re.compile(r"\bthreatened\s+by\b.+\band\s+live(?:s)?\s+in\b", re.I),
        ("multi", "threat_and_habitat",
         "Multi-hop: filter species by both threat AND habitat. "
         "Use find_species_by_threat_and_habitat."),
    ),
    (
        re.compile(r"\bwhich\s+\w[\w\s\-]*?\s+species\s+are\s+threatened\s+by\b", re.I),
        ("multi", "threat_and_status",
         "Multi-hop: filter species by both status AND threat. "
         "Use find_species_by_threat_and_status."),
    ),
    (
        re.compile(r"\b(?:conservation\s+)?actions\s+(?:are\s+)?taken\s+to\s+protect\s+species\s+threatened\s+by\b", re.I),
        ("multi", "actions_for_threat",
         "Multi-hop: list conservation actions for species facing a given threat. "
         "Use get_conservation_actions_for_threat."),
    ),

    # ---- single-hop ----
    (
        re.compile(r"\bconservation\s+status\s+of\b", re.I),
        ("single", "species_status",
         "Single-hop: use get_species_status."),
    ),
    (
        re.compile(r"\bscientific\s+name\s+of\b", re.I),
        ("single", "scientific_name",
         "Single-hop: use get_species_scientific_name."),
    ),
    (
        re.compile(r"\bwhere\s+does\s+\b.+\blive\b", re.I),
        ("single", "habitats_of_species",
         "Single-hop: use get_habitats_of_species."),
    ),
    (
        re.compile(r"\bwhat\s+threats?\s+does\b.+\bface\b", re.I),
        ("single", "threats_of_species",
         "Single-hop: use get_threats_of_species."),
    ),
    (
        re.compile(r"\bconservation\s+actions\s+are\s+protecting\b", re.I),
        ("single", "conservation_actions_of_species",
         "Single-hop: use get_conservation_actions_of_species."),
    ),
    (
        re.compile(r"\bwhich\s+species\s+are\s+threatened\s+by\b", re.I),
        ("single", "find_by_threat",
         "Single-hop: use find_species_by_threat."),
    ),
    (
        re.compile(r"\bwhich\s+species\s+live\s+in\b", re.I),
        ("single", "find_by_habitat",
         "Single-hop: use find_species_by_habitat."),
    ),
    (
        re.compile(r"\bwhich\s+species\s+are\s+listed\s+as\b", re.I),
        ("single", "find_by_status",
         "Single-hop: use find_species_by_status."),
    ),
]


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def regex_plan(question: str) -> Optional[Plan]:
    """Return a Plan from regex matching, or None if nothing matched."""
    for pattern, (hop, category, hint) in _PATTERNS:
        if pattern.search(question):
            return Plan(hop=hop, category=category, hint=hint, backend="regex")
    return None


def llm_plan(question: str, llm: Any) -> Plan:
    """One-call LLM classifier. Cheap fallback (~80–120 tokens)."""
    prompt = (
        "Classify the following question about endangered species into a "
        "JSON plan with keys 'hop' and 'category'.\n"
        f"Allowed hop: single | multi\n"
        f"Allowed category: {', '.join(CATEGORIES[:-1])}\n"
        "Output ONLY the JSON object — no prose, no markdown.\n\n"
        f"Question: {question}"
    )
    resp = llm.invoke(prompt)
    text = (resp.content or "").strip() if hasattr(resp, "content") else str(resp)
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in LLM plan response: {text!r}")
    data = json.loads(match.group(0))
    hop = data.get("hop")
    category = data.get("category", "unknown")
    if hop not in ("single", "multi") or category not in CATEGORIES:
        raise ValueError(f"LLM returned out-of-vocab plan: {data!r}")
    return Plan(
        hop=hop,
        category=category,
        hint=f"LLM-predicted plan: {hop}-hop, category={category}.",
        backend="llm",
    )


def plan(question: str,
         *,
         llm_fallback: Any = None,
         on_unknown: str = "default") -> Plan:
    """Return a :class:`Plan` for the question.

    Parameters
    ----------
    question : str
        Natural-language question.
    llm_fallback : Any, optional
        A LangChain-style LLM with ``.invoke(prompt) -> Message``. If
        provided, used only when the regex finds no match.
    on_unknown : str
        Behaviour when both regex and LLM fail: ``"default"`` returns a
        no-op plan, ``"raise"`` raises a ``ValueError``.
    """
    p = regex_plan(question)
    if p is not None:
        return p

    if llm_fallback is not None:
        try:
            return llm_plan(question, llm_fallback)
        except Exception:  # pragma: no cover - LLM may misbehave
            pass

    if on_unknown == "raise":
        raise ValueError(f"could not classify question: {question!r}")

    return Plan(
        hop="single",
        category="unknown",
        hint=("No specific plan inferred. Use the most specific tool that "
              "matches the question. Combine multiple tools for multi-hop "
              "questions."),
        backend="default",
    )
