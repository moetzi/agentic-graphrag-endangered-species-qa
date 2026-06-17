"""
Reference tool catalog used by the ground-truth generator and the evaluator.

The agent in graphrag_qa.py only exposes 2 tools today; this file enumerates a
fuller set of schema-aware parametric tools that the ground truth assumes.
Each entry maps a tool name to:
  - cypher: parameterised Cypher template the tool would execute
  - params: ordered list of parameter names
  - description: short intent (used as the tool docstring when bound to the LLM)

When you wire these into graphrag_qa.py, keep the names identical so the
"expected_tool" field in ground_truth.json remains comparable.
"""

TOOL_CATALOG = {
    # ---- single-hop, anchored on one species ---------------------------------
    "get_species_status": {
        "params": ["species_name"],
        "description": "Return the IUCN-style conservation status of a single species.",
        "cypher": (
            "MATCH (s:Species {name: $species_name}) "
            "RETURN s.status AS status"
        ),
    },
    "get_species_scientific_name": {
        "params": ["species_name"],
        "description": "Return the scientific (Latin) name of a single species.",
        "cypher": (
            "MATCH (s:Species {name: $species_name}) "
            "RETURN s.scientific_name AS scientific_name"
        ),
    },
    "get_species_description": {
        "params": ["species_name"],
        "description": "Return the descriptive paragraph for a single species.",
        "cypher": (
            "MATCH (s:Species {name: $species_name}) "
            "RETURN s.description AS description"
        ),
    },
    "get_habitats_of_species": {
        "params": ["species_name"],
        "description": "List habitats where a given species lives.",
        "cypher": (
            "MATCH (s:Species {name: $species_name})-[:LIVES_IN]->(h:Habitat) "
            "RETURN h.name AS habitat"
        ),
    },
    "get_threats_of_species": {
        "params": ["species_name"],
        "description": "List threats faced by a given species.",
        "cypher": (
            "MATCH (s:Species {name: $species_name})-[:THREATENED_BY]->(t:Threat) "
            "RETURN t.name AS threat"
        ),
    },
    "get_conservation_actions_of_species": {
        "params": ["species_name"],
        "description": "List conservation actions that protect a given species.",
        "cypher": (
            "MATCH (s:Species {name: $species_name})-[:PROTECTED_BY]->(a:ConservationAction) "
            "RETURN a.name AS action"
        ),
    },

    # ---- single-hop, anchored on a property/relation -------------------------
    "find_species_by_status": {
        "params": ["status"],
        "description": "Find species whose conservation status matches (case-insensitive contains).",
        "cypher": (
            "MATCH (s:Species) "
            "WHERE toLower(s.status) CONTAINS toLower($status) "
            "RETURN DISTINCT s.name AS species"
        ),
    },
    "find_species_by_threat": {
        "params": ["threat_keyword"],
        "description": "Find species threatened by a given threat keyword.",
        "cypher": (
            "MATCH (s:Species)-[:THREATENED_BY]->(t:Threat) "
            "WHERE toLower(t.name) CONTAINS toLower($threat_keyword) "
            "RETURN DISTINCT s.name AS species"
        ),
    },
    "find_species_by_habitat": {
        "params": ["habitat_keyword"],
        "description": "Find species that live in habitats matching the keyword.",
        "cypher": (
            "MATCH (s:Species)-[:LIVES_IN]->(h:Habitat) "
            "WHERE toLower(h.name) CONTAINS toLower($habitat_keyword) "
            "RETURN DISTINCT s.name AS species"
        ),
    },
    "find_species_by_conservation_action": {
        "params": ["action_keyword"],
        "description": "Find species protected by a given conservation action keyword.",
        "cypher": (
            "MATCH (s:Species)-[:PROTECTED_BY]->(a:ConservationAction) "
            "WHERE toLower(a.name) CONTAINS toLower($action_keyword) "
            "RETURN DISTINCT s.name AS species"
        ),
    },

    # ---- multi-hop -----------------------------------------------------------
    "get_neighbors_of_species": {
        "params": ["species_name"],
        "description": "Find species that share a habitat with the target species.",
        "cypher": (
            "MATCH (s:Species {name: $species_name})-[:SHARES_HABITAT_WITH]->(n:Species) "
            "RETURN DISTINCT n.name AS neighbor"
        ),
    },
    "get_neighbors_by_threat": {
        "params": ["species_name", "threat_keyword"],
        "description": "Among neighbours that share a habitat with the target species, return those facing a given threat.",
        "cypher": (
            "MATCH (s:Species {name: $species_name})-[:SHARES_HABITAT_WITH]->(n:Species) "
            "MATCH (n)-[:THREATENED_BY]->(t:Threat) "
            "WHERE toLower(t.name) CONTAINS toLower($threat_keyword) "
            "RETURN DISTINCT n.name AS neighbor"
        ),
    },
    "get_threats_of_neighbors": {
        "params": ["species_name"],
        "description": "Aggregate the distinct threats faced by neighbours that share a habitat with the target species.",
        "cypher": (
            "MATCH (:Species {name: $species_name})-[:SHARES_HABITAT_WITH]->(n:Species) "
            "MATCH (n)-[:THREATENED_BY]->(t:Threat) "
            "RETURN DISTINCT t.name AS threat"
        ),
    },
    "get_common_threats": {
        "params": ["species_a", "species_b"],
        "description": "Return the threats faced by both species_a and species_b.",
        "cypher": (
            "MATCH (a:Species {name: $species_a})-[:THREATENED_BY]->(t:Threat) "
            "MATCH (b:Species {name: $species_b})-[:THREATENED_BY]->(t) "
            "RETURN DISTINCT t.name AS threat"
        ),
    },
    "find_species_by_threat_and_habitat": {
        "params": ["threat_keyword", "habitat_keyword"],
        "description": "Find species threatened by a keyword AND living in a habitat matching another keyword.",
        "cypher": (
            "MATCH (s:Species)-[:THREATENED_BY]->(t:Threat) "
            "MATCH (s)-[:LIVES_IN]->(h:Habitat) "
            "WHERE toLower(t.name) CONTAINS toLower($threat_keyword) "
            "  AND toLower(h.name) CONTAINS toLower($habitat_keyword) "
            "RETURN DISTINCT s.name AS species"
        ),
    },
    "find_species_by_threat_and_status": {
        "params": ["threat_keyword", "status"],
        "description": "Find species threatened by a keyword AND with a given conservation status.",
        "cypher": (
            "MATCH (s:Species)-[:THREATENED_BY]->(t:Threat) "
            "WHERE toLower(t.name) CONTAINS toLower($threat_keyword) "
            "  AND toLower(s.status) CONTAINS toLower($status) "
            "RETURN DISTINCT s.name AS species"
        ),
    },
    "get_conservation_actions_for_threat": {
        "params": ["threat_keyword"],
        "description": "List conservation actions that protect species facing a given threat.",
        "cypher": (
            "MATCH (s:Species)-[:THREATENED_BY]->(t:Threat) "
            "MATCH (s)-[:PROTECTED_BY]->(a:ConservationAction) "
            "WHERE toLower(t.name) CONTAINS toLower($threat_keyword) "
            "RETURN DISTINCT a.name AS action"
        ),
    },
    "find_habitats_for_threat": {
        "params": ["threat_keyword"],
        "description": "List habitats where species facing a given threat live.",
        "cypher": (
            "MATCH (s:Species)-[:THREATENED_BY]->(t:Threat) "
            "MATCH (s)-[:LIVES_IN]->(h:Habitat) "
            "WHERE toLower(t.name) CONTAINS toLower($threat_keyword) "
            "RETURN DISTINCT h.name AS habitat"
        ),
    },
}
