"""
Generate the ground-truth QA dataset for the agentic GraphRAG evaluation.

Why a generator (instead of a hand-written JSON)?
    The graph is built from `data/species.json`. By deriving the gold answers
    from the same source we avoid drift: if a threat is added/removed for a
    species, the corresponding gold answer updates next time you regenerate.

Usage
-----
    python -m evaluation.build_ground_truth          # writes ground_truth.json
    python -m evaluation.build_ground_truth --check  # rebuilds and asserts
                                                     # the file is up to date

The schema of every QA item is documented in evaluation/README.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.tool_catalog import TOOL_CATALOG

ROOT = Path(__file__).resolve().parent.parent
SPECIES_PATH = ROOT / "data" / "species.json"
OUT_PATH = ROOT / "evaluation" / "ground_truth.json"


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def load_species() -> dict[str, dict]:
    raw = json.loads(SPECIES_PATH.read_text(encoding="utf-8"))
    return {entry["species"]: entry for entry in raw}


def tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Build an expected-tool record and validate it against TOOL_CATALOG."""
    if name not in TOOL_CATALOG:
        raise KeyError(f"Unknown tool '{name}'. Update tool_catalog.py first.")
    spec = TOOL_CATALOG[name]
    missing = [p for p in spec["params"] if p not in args]
    if missing:
        raise ValueError(f"Tool '{name}' is missing args: {missing}")
    return {"name": name, "args": args, "cypher": spec["cypher"]}


def context_tpl(template: str, **kwargs: Any) -> str:
    return template.format(**kwargs)


# --------------------------------------------------------------------------- #
# Per-category builders                                                       #
# --------------------------------------------------------------------------- #
def qa_status(species_name: str, db: dict[str, dict], qid: str) -> dict:
    s = db[species_name]
    return {
        "id": qid,
        "hop": "single",
        "category": "species_status",
        "difficulty": "easy",
        "question": f"What is the conservation status of the {species_name}?",
        "answer_type": "string",
        "gold_answer": s["status"],
        "gold_entities": [species_name, s["status"]],
        "gold_contexts": [f"{species_name} has conservation status: {s['status']}."],
        "expected_tools": [tool_call("get_species_status",
                                     {"species_name": species_name})],
    }


def qa_scientific_name(species_name: str, db: dict[str, dict], qid: str) -> dict:
    s = db[species_name]
    return {
        "id": qid,
        "hop": "single",
        "category": "scientific_name",
        "difficulty": "easy",
        "question": f"What is the scientific name of the {species_name}?",
        "answer_type": "string",
        "gold_answer": s["scientific_name"],
        "gold_entities": [species_name, s["scientific_name"]],
        "gold_contexts": [f"{species_name} has scientific name: {s['scientific_name']}."],
        "expected_tools": [tool_call("get_species_scientific_name",
                                     {"species_name": species_name})],
    }


def qa_habitats(species_name: str, db: dict[str, dict], qid: str) -> dict:
    s = db[species_name]
    habitats = sorted(s.get("habitats", []))
    assert habitats, f"{species_name} has no habitats"
    return {
        "id": qid,
        "hop": "single",
        "category": "habitats_of_species",
        "difficulty": "easy",
        "question": f"Where does the {species_name} live?",
        "answer_type": "set",
        "gold_answer": habitats,
        "gold_entities": [species_name] + habitats,
        "gold_contexts": [f"{species_name} lives in {h}." for h in habitats],
        "expected_tools": [tool_call("get_habitats_of_species",
                                     {"species_name": species_name})],
    }


def qa_threats(species_name: str, db: dict[str, dict], qid: str) -> dict:
    s = db[species_name]
    threats = sorted(s.get("threats", []))
    assert threats, f"{species_name} has no threats"
    return {
        "id": qid,
        "hop": "single",
        "category": "threats_of_species",
        "difficulty": "easy",
        "question": f"What threats does the {species_name} face?",
        "answer_type": "set",
        "gold_answer": threats,
        "gold_entities": [species_name] + threats,
        "gold_contexts": [f"{species_name} is threatened by {t}." for t in threats],
        "expected_tools": [tool_call("get_threats_of_species",
                                     {"species_name": species_name})],
    }


def qa_conservation_actions(species_name: str, db: dict[str, dict], qid: str) -> dict:
    s = db[species_name]
    actions = sorted(s.get("conservation_actions", []))
    assert actions, f"{species_name} has no conservation actions"
    return {
        "id": qid,
        "hop": "single",
        "category": "conservation_actions_of_species",
        "difficulty": "medium",
        "question": f"What conservation actions are protecting the {species_name}?",
        "answer_type": "set",
        "gold_answer": actions,
        "gold_entities": [species_name] + actions,
        "gold_contexts": [f"{species_name} is protected by: {a}." for a in actions],
        "expected_tools": [tool_call("get_conservation_actions_of_species",
                                     {"species_name": species_name})],
    }


def qa_find_by_threat(threat_keyword: str, db: dict[str, dict], qid: str) -> dict:
    kw = threat_keyword.lower()
    matched = sorted({
        name for name, s in db.items()
        if any(kw in t.lower() for t in s.get("threats", []))
    })
    assert matched, f"No species threatened by '{threat_keyword}'"
    return {
        "id": qid,
        "hop": "single",
        "category": "find_by_threat",
        "difficulty": "medium",
        "question": f"Which species are threatened by {threat_keyword}?",
        "answer_type": "set",
        "gold_answer": matched,
        "gold_entities": matched + [threat_keyword],
        "gold_contexts": [
            f"{name} is threatened by {t}."
            for name in matched
            for t in db[name]["threats"]
            if kw in t.lower()
        ],
        "expected_tools": [tool_call("find_species_by_threat",
                                     {"threat_keyword": threat_keyword})],
    }


def qa_find_by_habitat(habitat_keyword: str, db: dict[str, dict], qid: str) -> dict:
    kw = habitat_keyword.lower()
    matched = sorted({
        name for name, s in db.items()
        if any(kw in h.lower() for h in s.get("habitats", []))
    })
    assert matched, f"No species in habitat '{habitat_keyword}'"
    return {
        "id": qid,
        "hop": "single",
        "category": "find_by_habitat",
        "difficulty": "medium",
        "question": f"Which species live in {habitat_keyword}?",
        "answer_type": "set",
        "gold_answer": matched,
        "gold_entities": matched + [habitat_keyword],
        "gold_contexts": [
            f"{name} lives in {h}."
            for name in matched
            for h in db[name]["habitats"]
            if kw in h.lower()
        ],
        "expected_tools": [tool_call("find_species_by_habitat",
                                     {"habitat_keyword": habitat_keyword})],
    }


def qa_find_by_status(status: str, db: dict[str, dict], qid: str) -> dict:
    kw = status.lower()
    matched = sorted({
        name for name, s in db.items()
        if kw in (s.get("status") or "").lower()
    })
    assert matched, f"No species with status '{status}'"
    return {
        "id": qid,
        "hop": "single",
        "category": "find_by_status",
        "difficulty": "medium",
        "question": f"Which species are listed as {status}?",
        "answer_type": "set",
        "gold_answer": matched,
        "gold_entities": matched + [status],
        "gold_contexts": [f"{name} has conservation status: {db[name]['status']}." for name in matched],
        "expected_tools": [tool_call("find_species_by_status", {"status": status})],
    }


# ---- multi-hop ------------------------------------------------------------- #
def _resolved_neighbours(species_name: str, db: dict[str, dict]) -> list[str]:
    """Return shares_habitat_with entries that exist as Species nodes."""
    return sorted([
        n for n in db[species_name].get("shares_habitat_with") or []
        if n in db
    ])


def qa_neighbours(species_name: str, db: dict[str, dict], qid: str) -> dict:
    neighbours = _resolved_neighbours(species_name, db)
    assert neighbours, f"{species_name} has no resolvable neighbours"
    return {
        "id": qid,
        "hop": "multi",
        "category": "neighbours_of_species",
        "difficulty": "medium",
        "question": f"Which species share a habitat with the {species_name}?",
        "answer_type": "set",
        "gold_answer": neighbours,
        "gold_entities": [species_name] + neighbours,
        "gold_contexts": [f"{species_name} shares habitat with {n}." for n in neighbours],
        "expected_tools": [tool_call("get_neighbors_of_species",
                                     {"species_name": species_name})],
    }


def qa_common_threats(species_a: str, species_b: str,
                      db: dict[str, dict], qid: str) -> dict:
    a = set(db[species_a].get("threats") or [])
    b = set(db[species_b].get("threats") or [])
    common = sorted(a & b)
    assert common, f"No common threats between {species_a} and {species_b}"
    return {
        "id": qid,
        "hop": "multi",
        "category": "common_threats",
        "difficulty": "hard",
        "question": (f"Which threats are faced by both the {species_a} and "
                     f"the {species_b}?"),
        "answer_type": "set",
        "gold_answer": common,
        "gold_entities": [species_a, species_b] + common,
        "gold_contexts": (
            [f"{species_a} is threatened by {t}." for t in common] +
            [f"{species_b} is threatened by {t}." for t in common]
        ),
        "expected_tools": [tool_call("get_common_threats",
                                     {"species_a": species_a,
                                      "species_b": species_b})],
        "alternative_tools": [
            [tool_call("get_threats_of_species", {"species_name": species_a}),
             tool_call("get_threats_of_species", {"species_name": species_b})]
        ],
    }


def qa_neighbours_by_threat(species_name: str, threat_keyword: str,
                            db: dict[str, dict], qid: str) -> dict:
    kw = threat_keyword.lower()
    neighbours = _resolved_neighbours(species_name, db)
    matched = sorted([
        n for n in neighbours
        if any(kw in t.lower() for t in db[n].get("threats", []))
    ])
    assert matched, (f"No neighbours of {species_name} threatened by "
                     f"'{threat_keyword}'")
    return {
        "id": qid,
        "hop": "multi",
        "category": "neighbours_by_threat",
        "difficulty": "hard",
        "question": (f"Which species share a habitat with the {species_name} "
                     f"and are also threatened by {threat_keyword}?"),
        "answer_type": "set",
        "gold_answer": matched,
        "gold_entities": [species_name, threat_keyword] + matched,
        "gold_contexts": (
            [f"{species_name} shares habitat with {n}." for n in matched] +
            [f"{n} is threatened by {t}."
             for n in matched
             for t in db[n]["threats"] if kw in t.lower()]
        ),
        "expected_tools": [tool_call("get_neighbors_by_threat",
                                     {"species_name": species_name,
                                      "threat_keyword": threat_keyword})],
    }


def qa_threat_and_habitat(threat_keyword: str, habitat_keyword: str,
                          db: dict[str, dict], qid: str) -> dict:
    tk, hk = threat_keyword.lower(), habitat_keyword.lower()
    matched = sorted({
        name for name, s in db.items()
        if any(tk in t.lower() for t in s.get("threats", []))
        and any(hk in h.lower() for h in s.get("habitats", []))
    })
    assert matched, (f"No species match threat='{threat_keyword}' & "
                     f"habitat='{habitat_keyword}'")
    return {
        "id": qid,
        "hop": "multi",
        "category": "threat_and_habitat",
        "difficulty": "hard",
        "question": (f"Which species are threatened by {threat_keyword} and "
                     f"live in {habitat_keyword}?"),
        "answer_type": "set",
        "gold_answer": matched,
        "gold_entities": matched + [threat_keyword, habitat_keyword],
        "gold_contexts": [
            f"{n} is threatened by {t} and lives in {h}."
            for n in matched
            for t in db[n]["threats"] if tk in t.lower()
            for h in db[n]["habitats"] if hk in h.lower()
        ],
        "expected_tools": [tool_call("find_species_by_threat_and_habitat",
                                     {"threat_keyword": threat_keyword,
                                      "habitat_keyword": habitat_keyword})],
    }


def qa_threat_and_status(threat_keyword: str, status: str,
                         db: dict[str, dict], qid: str) -> dict:
    tk, sk = threat_keyword.lower(), status.lower()
    matched = sorted({
        name for name, s in db.items()
        if any(tk in t.lower() for t in s.get("threats", []))
        and sk in (s.get("status") or "").lower()
    })
    assert matched, (f"No species match threat='{threat_keyword}' & "
                     f"status='{status}'")
    return {
        "id": qid,
        "hop": "multi",
        "category": "threat_and_status",
        "difficulty": "hard",
        "question": (f"Which {status} species are threatened by "
                     f"{threat_keyword}?"),
        "answer_type": "set",
        "gold_answer": matched,
        "gold_entities": matched + [threat_keyword, status],
        "gold_contexts": [
            f"{n} is {db[n]['status']} and threatened by {t}."
            for n in matched
            for t in db[n]["threats"] if tk in t.lower()
        ],
        "expected_tools": [tool_call("find_species_by_threat_and_status",
                                     {"threat_keyword": threat_keyword,
                                      "status": status})],
    }


def qa_actions_for_threat(threat_keyword: str,
                          db: dict[str, dict], qid: str) -> dict:
    kw = threat_keyword.lower()
    actions: set[str] = set()
    contexts: list[str] = []
    for name, s in db.items():
        if any(kw in t.lower() for t in s.get("threats", [])):
            for a in s.get("conservation_actions", []) or []:
                actions.add(a)
                contexts.append(f"{name} (threatened by {threat_keyword}) "
                                f"is protected by: {a}.")
    matched = sorted(actions)
    assert matched, f"No actions found for threat '{threat_keyword}'"
    return {
        "id": qid,
        "hop": "multi",
        "category": "actions_for_threat",
        "difficulty": "hard",
        "question": (f"What conservation actions are taken to protect species "
                     f"threatened by {threat_keyword}?"),
        "answer_type": "set",
        "gold_answer": matched,
        "gold_entities": matched + [threat_keyword],
        "gold_contexts": contexts,
        "expected_tools": [tool_call("get_conservation_actions_for_threat",
                                     {"threat_keyword": threat_keyword})],
    }


# --------------------------------------------------------------------------- #
# Build the dataset (max 30, 18 single-hop + 12 multi-hop)                    #
# --------------------------------------------------------------------------- #
def build(db: dict[str, dict]) -> list[dict]:
    items: list[dict] = []

    # ---- 18 single-hop --------------------------------------------------- #
    items.append(qa_status("Sumatran orangutan", db, "sh-001"))
    items.append(qa_status("Vaquita", db, "sh-002"))
    items.append(qa_status("Snow leopard", db, "sh-003"))
    items.append(qa_status("Whale shark", db, "sh-004"))

    items.append(qa_scientific_name("Cheetah", db, "sh-005"))
    items.append(qa_scientific_name("Polar bear", db, "sh-006"))
    items.append(qa_scientific_name("Giant panda", db, "sh-007"))

    items.append(qa_habitats("Snow leopard", db, "sh-008"))
    items.append(qa_habitats("Polar bear", db, "sh-009"))
    items.append(qa_habitats("Monarch butterfly", db, "sh-010"))

    items.append(qa_threats("Sumatran orangutan", db, "sh-011"))
    items.append(qa_threats("Vaquita", db, "sh-012"))
    items.append(qa_threats("Monarch butterfly", db, "sh-013"))

    items.append(qa_conservation_actions("Yellowfin tuna", db, "sh-014"))
    items.append(qa_conservation_actions("Black-footed ferret", db, "sh-015"))

    items.append(qa_find_by_threat("poaching", db, "sh-016"))
    items.append(qa_find_by_habitat("Arctic", db, "sh-017"))
    items.append(qa_find_by_status("Critically Endangered", db, "sh-018"))

    # ---- 12 multi-hop ---------------------------------------------------- #
    # Neighbour queries: only species whose shares_habitat_with entries
    # actually resolve to other Species nodes in the graph.
    items.append(qa_neighbours("Sumatran orangutan", db, "mh-001"))
    items.append(qa_neighbours("Sunda tiger", db, "mh-002"))
    items.append(qa_neighbours("Vaquita", db, "mh-003"))
    items.append(qa_neighbours("Bornean elephant", db, "mh-004"))

    items.append(qa_common_threats("Sumatran orangutan", "Sumatran elephant",
                                   db, "mh-005"))
    items.append(qa_common_threats("Beluga whale", "Narwhal", db, "mh-006"))
    items.append(qa_common_threats("Bornean orangutan", "Sumatran orangutan",
                                   db, "mh-007"))

    items.append(qa_neighbours_by_threat("Sunda tiger", "habitat", db,
                                         "mh-008"))

    items.append(qa_threat_and_habitat("poaching", "Africa", db, "mh-009"))
    items.append(qa_threat_and_status("climate change", "Endangered", db,
                                      "mh-010"))

    items.append(qa_actions_for_threat("poaching", db, "mh-011"))
    items.append(qa_actions_for_threat("bycatch", db, "mh-012"))

    return items


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="Fail if ground_truth.json is out of date.")
    args = parser.parse_args()

    db = load_species()
    items = build(db)

    # Sanity invariants
    assert len(items) == 30, f"expected 30 items, got {len(items)}"
    assert sum(1 for i in items if i["hop"] == "single") == 18
    assert sum(1 for i in items if i["hop"] == "multi") == 12
    assert len({i["id"] for i in items}) == 30, "duplicate ids"

    payload = {
        "version": "1.0.0",
        "source": "data/species.json",
        "schema": {
            "node_labels": ["Species", "Habitat", "Threat", "ConservationAction"],
            "relationships": ["LIVES_IN", "THREATENED_BY", "PROTECTED_BY",
                              "SHARES_HABITAT_WITH"],
        },
        "counts": {
            "total": len(items),
            "single_hop": 18,
            "multi_hop": 12,
        },
        "items": items,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        existing = OUT_PATH.read_text(encoding="utf-8") if OUT_PATH.exists() else ""
        if existing != rendered:
            raise SystemExit("ground_truth.json is stale, "
                             "rerun: python -m evaluation.build_ground_truth")
        print("ground_truth.json is up to date.")
        return

    OUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUT_PATH.relative_to(ROOT)} "
          f"({len(items)} items: 18 single-hop, 12 multi-hop)")


if __name__ == "__main__":
    main()
