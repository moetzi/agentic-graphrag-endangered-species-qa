"""
Preflight connectivity check for the agentic GraphRAG QA stack.

Runs a sequence of checks:

  1. .env loaded and required variables present.
  2. Neo4j Bolt reachable + credentials accepted.
  3. Graph schema sniff (informational): counts of Species/Habitat/Threat/etc.
  4. Ollama HTTP endpoint reachable.
  5. Configured OLLAMA_MODEL is pulled locally.
  6. Tiny end-to-end sanity call: Ollama answers "ping".

Exit code is the number of failed checks (0 == all good). Informational
checks (like "graph has data") never fail the script — they only WARN — so
you can run this before or after ingestion.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable

from dotenv import find_dotenv, load_dotenv

# Coloured output if the terminal supports it; degrade gracefully on Windows
# cmd without ANSI.
try:
    import colorama  # type: ignore
    colorama.just_fix_windows_console()
    _COLOR = True
except Exception:
    _COLOR = False


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _COLOR else s


PASS = _c("32", "PASS")
FAIL = _c("31", "FAIL")
WARN = _c("33", "WARN")
INFO = _c("36", "INFO")


# --------------------------------------------------------------------------- #
# Test runner                                                                 #
# --------------------------------------------------------------------------- #
class CheckResult:
    __slots__ = ("name", "status", "detail", "duration_ms")

    def __init__(self, name: str, status: str, detail: str, duration_ms: float):
        self.name = name
        self.status = status
        self.detail = detail
        self.duration_ms = duration_ms


def run_check(name: str, fn: Callable[[], str]) -> CheckResult:
    """Execute one check. Return PASS on plain return, WARN on the special
    `WarnSignal` exception, and FAIL on any other exception."""
    print(f"  ... {name}", end="", flush=True)
    t0 = time.perf_counter()
    try:
        detail = fn() or ""
        status = PASS
    except WarnSignal as w:
        detail = str(w)
        status = WARN
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        status = FAIL
    dur = (time.perf_counter() - t0) * 1000
    pad = max(0, 40 - len(name))
    print(f"\r  [{status}] {name}{' ' * pad}({dur:6.0f} ms)  {detail}")
    return CheckResult(name, status, detail, dur)


class WarnSignal(Exception):
    """Raise from a check function to flag a non-fatal warning."""


# --------------------------------------------------------------------------- #
# Individual checks                                                           #
# --------------------------------------------------------------------------- #
def check_env() -> str:
    load_dotenv(find_dotenv(), override=False)
    required = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"missing env vars: {missing}")
    # Optional but recommended
    optional = {
        "OLLAMA_MODEL":    os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11450"),
    }
    return f"NEO4J_URI={os.getenv('NEO4J_URI')}, model={optional['OLLAMA_MODEL']}"


def check_neo4j_connectivity() -> str:
    from neo4j import GraphDatabase
    from neo4j.exceptions import AuthError, ServiceUnavailable

    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    pw   = os.environ["NEO4J_PASSWORD"]

    try:
        driver = GraphDatabase.driver(uri, auth=(user, pw))
        try:
            # verify_connectivity tries a real handshake.
            driver.verify_connectivity()
            with driver.session() as sess:
                rec = sess.run("RETURN 1 AS ok").single()
            if not rec or rec["ok"] != 1:
                raise RuntimeError("RETURN 1 did not yield 1")
        finally:
            driver.close()
    except AuthError as exc:
        raise RuntimeError(f"auth rejected ({exc.code or 'no code'})") from exc
    except ServiceUnavailable as exc:
        raise RuntimeError(
            f"cannot reach {uri} — is `docker compose up -d` running and healthy?"
        ) from exc
    return f"connected to {uri} as {user}"


def check_neo4j_schema() -> str:
    """Informational only — surfaces ingestion progress."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        with driver.session() as s:
            counts = s.run(
                """
                CALL {
                    MATCH (n:Species)            RETURN count(n) AS species
                }
                CALL {
                    MATCH (n:Habitat)            RETURN count(n) AS habitats
                }
                CALL {
                    MATCH (n:Threat)             RETURN count(n) AS threats
                }
                CALL {
                    MATCH (n:ConservationAction) RETURN count(n) AS actions
                }
                CALL {
                    MATCH ()-[r:SHARES_HABITAT_WITH]->()
                    RETURN count(r) AS shares
                }
                RETURN species, habitats, threats, actions, shares
                """
            ).single()
            # s.run(...).single() can return None (e.g. empty result); coerce to zeros
            if counts is None:
                counts = {"species": 0, "habitats": 0, "threats": 0, "actions": 0, "shares": 0}
    finally:
        driver.close()

    msg = (f"Species={counts['species']} Habitat={counts['habitats']} "
           f"Threat={counts['threats']} Actions={counts['actions']} "
           f"SHARES_HABITAT_WITH={counts['shares']}")
    if counts["species"] == 0:
        raise WarnSignal(msg + "  (graph empty — run `python ingestion.py`)")
    if counts["shares"] == 0:
        raise WarnSignal(msg + "  (no SHARES_HABITAT_WITH yet — multi-hop "
                         "questions will return empty)")
    return msg


def check_ollama_reachable() -> str:
    import ollama
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11450")
    try:
        # ollama.Client.list() returns local models; raises on connection error
        client = ollama.Client(host=base)
        models = client.list()
    except Exception as exc:
        raise RuntimeError(
            f"cannot reach {base} — is `ollama serve` running?"
        ) from exc
    n = len(getattr(models, "models", []) or [])
    return f"{base} reachable — {n} model(s) installed"


def check_ollama_model() -> str:
    import ollama
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11450")
    wanted = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    client = ollama.Client(host=base)
    listing = client.list()

    # The python client returns objects with `.model` (newer) or dicts with
    # 'name' (older) — handle both.
    available = []
    for m in getattr(listing, "models", []) or []:
        name = getattr(m, "model", None) or getattr(m, "name", None) \
               or (m.get("model") if isinstance(m, dict) else None) \
               or (m.get("name")  if isinstance(m, dict) else None)
        if name:
            available.append(name)

    if not available:
        raise RuntimeError("no models installed — run "
                           f"`ollama pull {wanted}`")

    # Match either the exact tag the user requested or a tag whose family
    # matches (e.g., user has "llama3.1:latest", configured "llama3.1").
    match = next(
        (m for m in available
         if m == wanted or m.startswith(wanted + ":") or m.split(":", 1)[0] == wanted),
        None,
    )
    if not match:
        raise RuntimeError(f"model '{wanted}' not pulled. Have: "
                           f"{', '.join(available)}. "
                           f"Run `ollama pull {wanted}`.")
    return f"{match} available"


def check_ollama_chat() -> str:
    """Tiny end-to-end test: ask the model to answer 'ping' with a single
    word. Catches anything verify-only checks miss (e.g., model loaded but
    responding 500)."""
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage

    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11450")
    llm = ChatOllama(model=model, base_url=base, temperature=0)
    resp = llm.invoke([
        SystemMessage(content="Reply with exactly the word: pong"),
        HumanMessage(content="ping"),
    ])
    text = (str(resp.content) if resp.content else "").strip()
    if not text:
        raise RuntimeError("empty response from model")
    snippet = text[:60].replace("\n", " ")
    return f"model replied: {snippet!r}"


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    skip_chat = "--no-chat" in argv

    print(_c("1", "Endangered Species GraphRAG — connectivity check"))
    print()

    checks: list[tuple[str, Callable[[], str]]] = [
        ("environment variables",       check_env),
        ("Neo4j connectivity",          check_neo4j_connectivity),
        ("Neo4j schema (informational)", check_neo4j_schema),
        ("Ollama reachable",            check_ollama_reachable),
        ("Ollama model present",        check_ollama_model),
    ]
    if not skip_chat:
        checks.append(("Ollama chat round-trip", check_ollama_chat))

    results: list[CheckResult] = []
    for name, fn in checks:
        results.append(run_check(name, fn))

    print()
    failed = sum(1 for r in results if r.status == FAIL)
    warned = sum(1 for r in results if r.status == WARN)
    passed = len(results) - failed - warned
    print(f"  {passed} passed, {warned} warned, {failed} failed")

    if failed:
        print(f"\n{FAIL}: fix the issues above before running the evaluator.")
        return failed
    if warned:
        print(f"\n{WARN}: stack is reachable but not fully provisioned. "
              "Run `python ingestion.py` if the graph is empty.")
    else:
        print(f"\n{PASS}: ready to run. Try: "
              f"`python -m evaluation.runner --gold-tools`")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
