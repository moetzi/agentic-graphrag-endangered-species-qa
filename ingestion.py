"""
Ingest data/species.json into Neo4j.

Schema (must match evaluation/ground_truth.json):
    (:Species {name, scientific_name, status, description,
               population, weight, height, length})
    (:Habitat {name})
    (:Threat {name})
    (:ConservationAction {name})

    (:Species)-[:LIVES_IN]->(:Habitat)
    (:Species)-[:THREATENED_BY]->(:Threat)
    (:Species)-[:PROTECTED_BY]->(:ConservationAction)
    (:Species)-[:SHARES_HABITAT_WITH]->(:Species)   -- bidirectional

Run:
    docker compose up -d
    python ingestion.py
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase
from typing import cast, LiteralString

load_dotenv(find_dotenv())

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

DATA_FILE = Path(__file__).parent / "data" / "species.json"

# Stringify population/weight/height/length defensively: the source mixes ints,
# strings, and nulls (e.g., 13846 vs "1,000–1,800" vs null).
_STR_PROPS = ("population", "weight", "height", "length")


CONSTRAINTS = [
    "CREATE CONSTRAINT species_name IF NOT EXISTS "
    "FOR (s:Species) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT habitat_name IF NOT EXISTS "
    "FOR (h:Habitat) REQUIRE h.name IS UNIQUE",
    "CREATE CONSTRAINT threat_name IF NOT EXISTS "
    "FOR (t:Threat) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT action_name IF NOT EXISTS "
    "FOR (a:ConservationAction) REQUIRE a.name IS UNIQUE",
]


# Two-pass ingestion:
#   1. UPSERT every Species + its habitats / threats / conservation actions.
#   2. UPSERT SHARES_HABITAT_WITH only when both endpoints exist as Species.
#
# We do not auto-create Species nodes for shares_habitat_with entries that
# aren't in species.json (e.g., "Krill") so the graph stays tied to the source.
UPSERT_SPECIES_CYPHER = """
MERGE (s:Species {name: $name})
SET s.scientific_name = $scientific_name,
    s.status          = $status,
    s.description     = $description,
    s.population      = $population,
    s.weight          = $weight,
    s.height          = $height,
    s.length          = $length

FOREACH (h IN $habitats |
    MERGE (hab:Habitat {name: h})
    MERGE (s)-[:LIVES_IN]->(hab))

FOREACH (t IN $threats |
    MERGE (thr:Threat {name: t})
    MERGE (s)-[:THREATENED_BY]->(thr))

FOREACH (a IN $actions |
    MERGE (act:ConservationAction {name: a})
    MERGE (s)-[:PROTECTED_BY]->(act))
"""

# Bidirectional MERGE: sharing a habitat is symmetric. We write both directions
# so the agent's Cypher tools work regardless of which endpoint is queried.
UPSERT_SHARES_HABITAT_CYPHER = """
MATCH (a:Species {name: $species})
UNWIND $neighbors AS neighbor_name
MATCH (b:Species {name: neighbor_name})
MERGE (a)-[:SHARES_HABITAT_WITH]->(b)
MERGE (b)-[:SHARES_HABITAT_WITH]->(a)
"""


def _to_str(value) -> str | None:
    if value is None:
        return None
    return str(value)


def ingest_species_data(data_file: Path = DATA_FILE) -> dict:
    if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
        raise ValueError(
            "NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set in .env"
        )
    if not data_file.exists():
        raise FileNotFoundError(data_file)

    species_list = json.loads(data_file.read_text(encoding="utf-8"))
    species_names = {entry["species"] for entry in species_list}

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            # Constraints first — much faster MERGE afterwards.
            for stmt in CONSTRAINTS:
                # type-checker: session.run expects a LiteralString | Query;
                # cast the runtime str to LiteralString for correctness.
                session.run(cast(LiteralString, stmt))

            # Pass 1: species + habitats/threats/actions
            for entry in species_list:
                params = {
                    "name":            entry["species"],
                    "scientific_name": entry.get("scientific_name"),
                    "status":          entry.get("status"),
                    "description":     entry.get("description"),
                    "habitats":        entry.get("habitats", []) or [],
                    "threats":         entry.get("threats", []) or [],
                    "actions":         entry.get("conservation_actions", []) or [],
                }
                for k in _STR_PROPS:
                    params[k] = _to_str(entry.get(k))
                session.run(UPSERT_SPECIES_CYPHER, **params)

            # Pass 2: SHARES_HABITAT_WITH (only species-to-species edges).
            shares_edges = 0
            skipped_neighbors: list[tuple[str, str]] = []
            for entry in species_list:
                resolved, unresolved = [], []
                for n in entry.get("shares_habitat_with") or []:
                    (resolved if n in species_names else unresolved).append(n)
                if resolved:
                    session.run(
                        UPSERT_SHARES_HABITAT_CYPHER,
                        species=entry["species"],
                        neighbors=resolved,
                    )
                    shares_edges += len(resolved)
                for n in unresolved:
                    skipped_neighbors.append((entry["species"], n))

            result = session.run(
                """
                CALL {
                    MATCH (s:Species)             RETURN count(s) AS species
                }
                CALL {
                    MATCH (h:Habitat)             RETURN count(h) AS habitats
                }
                CALL {
                    MATCH (t:Threat)              RETURN count(t) AS threats
                }
                CALL {
                    MATCH (a:ConservationAction)  RETURN count(a) AS actions
                }
                CALL {
                    MATCH ()-[r:SHARES_HABITAT_WITH]->() RETURN count(r) AS shares_rel
                }
                RETURN species, habitats, threats, actions, shares_rel
                """
            )
            counts = result.single() or {}

        summary = {
            "species":             counts.get("species", 0),
            "habitats":            counts.get("habitats", 0),
            "threats":             counts.get("threats", 0),
            "actions":             counts.get("actions", 0),
            "shares_habitat_rels": counts.get("shares_rel", 0),   # 2x edges (bidir)
            "shares_pairs_added":  shares_edges,
            "skipped_neighbors":   skipped_neighbors,
        }
        return summary
    finally:
        driver.close()


if __name__ == "__main__":
    summary = ingest_species_data()
    print("Ingestion complete.")
    print(f"  Species:                  {summary['species']}")
    print(f"  Habitats:                 {summary['habitats']}")
    print(f"  Threats:                  {summary['threats']}")
    print(f"  ConservationActions:      {summary['actions']}")
    print(f"  SHARES_HABITAT_WITH rels: {summary['shares_habitat_rels']} "
          f"(bidirectional, {summary['shares_pairs_added']} ordered upserts)")
    if summary["skipped_neighbors"]:
        print(f"\n  Skipped {len(summary['skipped_neighbors'])} "
              f"shares_habitat_with entries (not present as Species nodes):")
        for src, neighbor in summary["skipped_neighbors"]:
            print(f"    - {src!r} -> {neighbor!r}")
