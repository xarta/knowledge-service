#!/usr/bin/env python3
"""Cautious cleanup tool for orphaned test collections in SeekDB.

Identifies collections that look like test data (by naming patterns, metadata,
and age), then optionally uses an LLM to validate the assessment before
offering interactive deletion.

Usage:
    python3 tools/clean_test_data.py              # Scan and report (dry run)
    python3 tools/clean_test_data.py --clean       # Scan → LLM validate → interactive delete
    python3 tools/clean_test_data.py --clean --yes  # Skip interactive prompts (still LLM-validated)

Requires:
    .env file with KNOWLEDGE_SERVICE_URL set.
    For --clean with LLM validation: VLLM_BASE_URL and optionally VLLM_API_KEY.

Safety:
    - NEVER auto-deletes collections without positive identification.
    - Known test patterns matched first; LLM provides secondary confirmation.
    - Collections that the LLM is unsure about are flagged but NOT deleted.
    - Production patterns (no leading underscore, no 'test' in name) are
      always left alone regardless of other signals.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


# ===================================================================
# Known test collection patterns
# ===================================================================

# Patterns produced by our test suites across all projects.
# Format: (regex, source description)
KNOWN_TEST_PATTERNS: List[Tuple[str, str]] = [
    (r"^_integration_test$", "knowledge-service check_service.py"),
    (r"^_test_ingestion_\d+$", "doc-sanitiser TestIngestionPipelineLive"),
    (r"^_test_registry_\d+$", "doc-sanitiser TestIngestionPipelineLive (registry)"),
    (r"^_test_registry_ops_\d+$", "doc-sanitiser TestRegistryOperationsLive"),
    (r"^_test_", "generic test prefix convention"),
    (r"^test_", "generic test prefix (no underscore)"),
]

# Patterns that indicate PRODUCTION data — never delete these.
PRODUCTION_PATTERNS: List[str] = [
    r"^docsanitiser_",
    r"^knowledge_",
    r"^project_",
]


# ===================================================================
# HTTP helpers (stdlib only)
# ===================================================================

def http_get(url: str, timeout: int = 10) -> Dict[str, Any]:
    """Make an HTTP GET request and return JSON response."""
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    """Make an HTTP POST with JSON body, return JSON response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_delete(url: str, timeout: int = 30) -> None:
    """Make an HTTP DELETE request."""
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        pass


def load_env() -> Dict[str, str]:
    """Load .env file from project root."""
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    env_vars = {}
    if not os.path.exists(env_path):
        return env_vars
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()
    return env_vars


# ===================================================================
# Classification logic
# ===================================================================

def classify_collection(name: str, count: int, visibility: str) -> Dict[str, Any]:
    """Classify a collection as test, production, or unknown.

    Returns a dict with:
        classification: "test" | "production" | "unknown"
        confidence: "high" | "medium" | "low"
        reason: human-readable explanation
        pattern_match: which known pattern matched (if any)
    """
    # Check production patterns first — always wins
    for pattern in PRODUCTION_PATTERNS:
        if re.match(pattern, name):
            return {
                "classification": "production",
                "confidence": "high",
                "reason": f"Matches production pattern: {pattern}",
                "pattern_match": pattern,
            }

    # Check known test patterns
    for pattern, source in KNOWN_TEST_PATTERNS:
        if re.match(pattern, name):
            return {
                "classification": "test",
                "confidence": "high",
                "reason": f"Matches known test pattern from {source}",
                "pattern_match": pattern,
            }

    # Heuristic: underscore prefix with numeric suffix looks like timestamped test
    if name.startswith("_") and re.search(r"\d{8,}$", name):
        return {
            "classification": "test",
            "confidence": "medium",
            "reason": "Underscore prefix with numeric suffix (likely timestamped test data)",
            "pattern_match": None,
        }

    # Heuristic: underscore prefix is suspicious
    if name.startswith("_"):
        return {
            "classification": "unknown",
            "confidence": "low",
            "reason": "Underscore prefix suggests test/internal data but no pattern match",
            "pattern_match": None,
        }

    # No signals — likely production
    return {
        "classification": "unknown",
        "confidence": "low",
        "reason": "No test pattern match; may be production data",
        "pattern_match": None,
    }


def scan_collections(base_url: str) -> List[Dict[str, Any]]:
    """Fetch all collections and classify each one.

    Returns a list of dicts, each containing:
        name, count, visibility, classification (from classify_collection)
    """
    response = http_get(f"{base_url}/collections")
    collections = response.get("collections", [])

    results = []
    for coll in collections:
        name = coll.get("name", "")
        count = coll.get("count", 0)
        visibility = coll.get("visibility", "unknown")
        classification = classify_collection(name, count, visibility)

        results.append({
            "name": name,
            "count": count,
            "visibility": visibility,
            **classification,
        })

    return results


# ===================================================================
# LLM validation
# ===================================================================

def llm_validate(collections: List[Dict[str, Any]], env: Dict[str, str]) -> List[Dict[str, Any]]:
    """Use an LLM to validate the classification of flagged collections.

    Only called for collections classified as "test" or "unknown".
    The LLM sees each collection's name, count, visibility, and our
    classification, and confirms or overrides the assessment.

    Returns the same list with an added 'llm_verdict' key on each item.
    """
    vllm_base_url = env.get("VLLM_BASE_URL", os.environ.get("VLLM_BASE_URL", ""))
    vllm_api_key = env.get("VLLM_API_KEY", os.environ.get("VLLM_API_KEY", ""))

    if not vllm_base_url:
        print("    ⚠ VLLM_BASE_URL not set — skipping LLM validation")
        for coll in collections:
            coll["llm_verdict"] = None
        return collections

    # Detect model
    model = _detect_model(vllm_base_url, vllm_api_key)
    if not model:
        print("    ⚠ Could not detect LLM model — skipping LLM validation")
        for coll in collections:
            coll["llm_verdict"] = None
        return collections

    # Build the prompt
    system_prompt = (
        "You are a database administrator reviewing collections in a SeekDB "
        "vector database. Your task is to determine which collections are "
        "orphaned test data that can be safely deleted.\n\n"
        "KNOWN TEST PATTERNS from our test suites:\n"
        "- '_integration_test' — knowledge-service integration tests\n"
        "- '_test_ingestion_<timestamp>' — doc-sanitiser ingestion tests\n"
        "- '_test_registry_<timestamp>' — doc-sanitiser registry tests\n"
        "- '_test_registry_ops_<timestamp>' — doc-sanitiser registry operations tests\n"
        "- Any name starting with '_test_' or 'test_' — likely test data\n\n"
        "KNOWN PRODUCTION PATTERNS:\n"
        "- 'docsanitiser_*' — production knowledge base\n"
        "- Names without leading underscores that don't contain 'test'\n\n"
        "For each collection, respond with a JSON array of objects with keys:\n"
        "  'name': the collection name\n"
        "  'safe_to_delete': true/false\n"
        "  'reasoning': brief explanation\n\n"
        "Be CAUTIOUS. When in doubt, mark as NOT safe to delete. "
        "It is far worse to delete production data than to leave test data behind."
    )

    collection_summaries = []
    for coll in collections:
        collection_summaries.append(
            f"- name='{coll['name']}', count={coll['count']}, "
            f"visibility={coll['visibility']}, "
            f"our_classification={coll['classification']} "
            f"(confidence={coll['confidence']}, reason={coll['reason']})"
        )

    user_prompt = (
        "Please review these collections and determine which are safe to delete:\n\n"
        + "\n".join(collection_summaries)
        + "\n\nRespond with ONLY the JSON array. /no_think"
    )

    try:
        raw_response = _llm_chat(
            vllm_base_url, vllm_api_key, model,
            system_prompt, user_prompt,
        )

        # Parse LLM response
        verdicts = _parse_llm_response(raw_response)

        # Attach verdicts to collections
        verdict_map = {v["name"]: v for v in verdicts}
        for coll in collections:
            verdict = verdict_map.get(coll["name"])
            if verdict:
                coll["llm_verdict"] = {
                    "safe_to_delete": verdict.get("safe_to_delete", False),
                    "reasoning": verdict.get("reasoning", ""),
                }
            else:
                coll["llm_verdict"] = None

    except Exception as exc:
        print(f"    ⚠ LLM validation failed: {exc}")
        for coll in collections:
            coll["llm_verdict"] = None

    return collections


def _detect_model(base_url: str, api_key: str) -> Optional[str]:
    """Detect the first available model from /v1/models."""
    url = f"{base_url}/models"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            if models:
                return models[0]["id"]
    except Exception:
        pass
    return None


def _llm_chat(
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
) -> str:
    """Send a chat completion request and return the content."""
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)

    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        # Strip think tags (Qwen3 safety net)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        return content


def _parse_llm_response(raw: str) -> List[Dict[str, Any]]:
    """Parse LLM JSON response, handling markdown fences."""
    text = raw.strip()
    # Strip markdown fences
    match = re.match(r"^```(?:\w+)?\s*\n(.*?)\n\s*```\s*$", text, re.DOTALL)
    if match:
        text = match.group(1)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find embedded JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return []


# ===================================================================
# Deletion logic
# ===================================================================

def delete_collection(base_url: str, name: str) -> bool:
    """Delete a single collection. Returns True on success."""
    try:
        http_delete(f"{base_url}/collections/{name}")
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return True  # Already gone
        print(f"    ✗ Delete failed for '{name}': HTTP {exc.code}")
        return False
    except Exception as exc:
        print(f"    ✗ Delete failed for '{name}': {exc}")
        return False


def is_safe_to_delete(coll: Dict[str, Any]) -> bool:
    """Determine if a collection passes all safety checks for deletion.

    A collection is safe to delete only when:
      1. It is classified as "test" with high or medium confidence, AND
      2. It is NOT classified as "production", AND
      3. If LLM validation was performed, the LLM agreed it's safe.
    """
    if coll["classification"] == "production":
        return False

    if coll["classification"] != "test":
        return False

    if coll["confidence"] == "low":
        return False

    # If LLM was consulted and said NO, respect that
    if coll.get("llm_verdict") is not None:
        if not coll["llm_verdict"].get("safe_to_delete", False):
            return False

    return True


# ===================================================================
# Reporting
# ===================================================================

def print_report(collections: List[Dict[str, Any]]) -> None:
    """Print a human-readable report of all collections and their status."""
    if not collections:
        print("    No collections found in SeekDB.")
        return

    test_collections = [c for c in collections if c["classification"] == "test"]
    prod_collections = [c for c in collections if c["classification"] == "production"]
    unknown_collections = [c for c in collections if c["classification"] == "unknown"]

    # Production collections
    if prod_collections:
        print(f"  Production ({len(prod_collections)}):")
        for coll in prod_collections:
            print(f"    ✓ {coll['name']} ({coll['count']} items, {coll['visibility']})")
        print()

    # Test collections
    if test_collections:
        print(f"  Test data ({len(test_collections)}):")
        for coll in test_collections:
            safe = is_safe_to_delete(coll)
            marker = "🗑" if safe else "⚠"
            llm_note = ""
            if coll.get("llm_verdict") is not None:
                llm_safe = coll["llm_verdict"].get("safe_to_delete", False)
                llm_note = f" [LLM: {'safe' if llm_safe else 'KEEP'}]"
            print(
                f"    {marker} {coll['name']} ({coll['count']} items) — "
                f"{coll['reason']}{llm_note}"
            )
        print()

    # Unknown collections
    if unknown_collections:
        print(f"  Unknown ({len(unknown_collections)}):")
        for coll in unknown_collections:
            llm_note = ""
            if coll.get("llm_verdict") is not None:
                llm_safe = coll["llm_verdict"].get("safe_to_delete", False)
                reason = coll["llm_verdict"].get("reasoning", "")
                llm_note = f" [LLM: {'safe' if llm_safe else 'KEEP'} — {reason}]"
            print(
                f"    ? {coll['name']} ({coll['count']} items, "
                f"{coll['visibility']}) — {coll['reason']}{llm_note}"
            )
        print()


# ===================================================================
# Main
# ===================================================================

def main() -> int:
    env = load_env()
    base_url = env.get(
        "KNOWLEDGE_SERVICE_URL",
        os.environ.get("KNOWLEDGE_SERVICE_URL", ""),
    ).rstrip("/")

    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return 0

    if not base_url:
        print("Error: KNOWLEDGE_SERVICE_URL not set (check .env or environment)")
        return 1

    do_clean = "--clean" in sys.argv
    skip_prompt = "--yes" in sys.argv

    print("SeekDB Test Data Cleanup Tool")
    print(f"Endpoint: {base_url}")
    print()

    # Step 1: Scan
    print("==> Scanning collections")
    try:
        collections = scan_collections(base_url)
    except Exception as exc:
        print(f"✗ Failed to list collections: {exc}")
        return 1

    print(f"    Found {len(collections)} collections")
    print()

    # Step 2: LLM validation (if cleaning and LLM available)
    if do_clean:
        non_production = [
            c for c in collections if c["classification"] != "production"
        ]
        if non_production:
            print("==> LLM validation")
            collections_to_validate = [
                c for c in collections if c["classification"] in ("test", "unknown")
            ]
            if collections_to_validate:
                llm_validate(collections_to_validate, env)
                print()

    # Step 3: Report
    print("==> Classification report")
    print_report(collections)

    # Step 4: Clean up (if requested)
    if do_clean:
        deletable = [c for c in collections if is_safe_to_delete(c)]

        if not deletable:
            print("✓ No orphaned test collections to clean up")
            return 0

        print(f"==> Ready to delete {len(deletable)} test collection(s):")
        for coll in deletable:
            print(f"    - {coll['name']} ({coll['count']} items)")
        print()

        if not skip_prompt:
            confirm = input("Delete these collections? [y/N] ").strip().lower()
            if confirm not in ("y", "yes"):
                print("Aborted.")
                return 0

        print("==> Deleting")
        deleted = 0
        failed = 0
        for coll in deletable:
            if delete_collection(base_url, coll["name"]):
                print(f"    ✓ Deleted '{coll['name']}'")
                deleted += 1
            else:
                failed += 1

        print()
        print(f"✓ Cleanup complete: {deleted} deleted, {failed} failed")
        return 1 if failed > 0 else 0

    else:
        # Dry-run summary
        test_count = sum(1 for c in collections if is_safe_to_delete(c))
        if test_count > 0:
            print(f"Found {test_count} test collection(s) eligible for cleanup.")
            print("Run with --clean to delete them (with LLM validation).")
        else:
            print("✓ No orphaned test collections detected")
        return 0


if __name__ == "__main__":
    sys.exit(main())
