#!/usr/bin/env python3
"""Health check and integration test tool for knowledge-service.

Usage:
    python3 tools/check_service.py          # Health check only
    python3 tools/check_service.py --test   # Run integration tests
    python3 tools/check_service.py --all    # Full test suite

Requires:
    .env file with KNOWLEDGE_SERVICE_URL set
"""

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict


def load_env() -> Dict[str, str]:
    """Load .env file from project root."""
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    env_vars = {}

    if not os.path.exists(env_path):
        print(f"Warning: .env file not found at {env_path}")
        return env_vars

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()

    return env_vars


def http_get(url: str, timeout: int =10) -> Dict[str, Any]:
    """Make an HTTP GET request and return JSON response."""
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    """Make an HTTP POST request and return JSON response."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"))
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_delete(url: str, timeout: int = 30) -> None:
    """Make an HTTP DELETE request."""
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        pass  # 204 No Content expected


def check_health(base_url: str) -> bool:
    """Check service health."""
    try:
        print("==> Checking root endpoint")
        root = http_get(base_url)
        print(f"    Service: {root.get('service')}")
        print(f"    Version: {root.get('version')}")
        print(f"    Status: {root.get('status')}")
        print()

        print("==> Checking health endpoint")
        health = http_get(f"{base_url}/health")
        print(f"    Status: {health.get('status')}")

        deps = health.get("dependencies", {})
        for name, info in deps.items():
            if isinstance(info, dict):
                connected = info.get("connected", info.get("status") == "not_configured")
                status_str = "✓" if connected else "✗"
                print(f"    {status_str} {name}: {info}")
            else:
                print(f"    - {name}: {info}")

        print()
        return health.get("status") in ("healthy", "degraded")

    except urllib.error.HTTPError as exc:
        print(f"✗ HTTP error: {exc.code} {exc.reason}")
        return False
    except urllib.error.URLError as exc:
        print(f"✗ Connection error: {exc.reason}")
        return False
    except Exception as exc:
        print(f"✗ Error: {exc}")
        return False


def test_ingestion(base_url: str) -> bool:
    """Test document ingestion into SeekDB and verify stored content."""
    print("==> Testing ingestion")

    test_collection = "_integration_test"
    payload = {
        "files": {
            "test/doc1.md": "# Test Document\n\nThis is a test document for the knowledge service.\n\nIt covers ingestion verification.",
            "test/doc2.md": "# Another Document\n\nThis document contains different test content about queries and retrieval.",
        },
        "collection_name": test_collection,
        "visibility": "public",
    }

    try:
        # Clean up any leftover test collection first
        try:
            http_delete(f"{base_url}/collections/{test_collection}")
            print("    (cleaned up previous test collection)")
        except Exception:
            pass  # Collection may not exist

        response = http_post(f"{base_url}/ingest", payload, timeout=120)
        chunks_ingested = response.get("chunks_ingested", 0)
        files_processed = response.get("files_processed", 0)
        print(f"    Collection: {response.get('collection')}")
        print(f"    Chunks ingested: {chunks_ingested}")
        print(f"    Files processed: {files_processed}")
        print(f"    Duration: {response.get('duration_seconds', 0):.2f}s")

        if chunks_ingested == 0:
            print("    ✗ No chunks ingested — pipeline may be broken")
            return False

        if files_processed != 2:
            print(f"    ✗ Expected 2 files processed, got {files_processed}")
            return False

        # Verify the collection exists and has the right count
        print("    Verifying SeekDB content...")
        collections_resp = http_get(f"{base_url}/collections")
        collections = collections_resp.get("collections", [])
        test_coll = next(
            (c for c in collections if c.get("name") == test_collection), None
        )
        if not test_coll:
            print(f"    ✗ Collection '{test_collection}' not found in SeekDB")
            return False

        stored_count = test_coll.get("count", 0)
        if stored_count != chunks_ingested:
            print(f"    ✗ Count mismatch: ingested {chunks_ingested}, stored {stored_count}")
            return False

        print(f"    ✓ SeekDB verified: {stored_count} chunks stored in '{test_collection}'")
        print()
        return True

    except urllib.error.HTTPError as exc:
        print(f"✗ HTTP error: {exc.code} {exc.reason}")
        body = exc.read().decode("utf-8") if exc.fp else ""
        if body:
            print(f"    Response: {body}")
        return False
    except Exception as exc:
        print(f"✗ Error: {exc}")
        return False


def test_query(base_url: str) -> bool:
    """Test RAG query against ingested test collection."""
    print("==> Testing query")

    test_collection = "_integration_test"
    payload = {
        "question": "What is this test document about?",
        "collections": [test_collection],
        "n_results": 5,
    }

    try:
        response = http_post(f"{base_url}/query", payload, timeout=60)
        total = response.get("total_results", 0)
        print(f"    Total results: {total}")

        if total == 0:
            print("    ✗ No results returned — retrieval may be broken")
            return False

        results = response.get("results", [])
        for i, result in enumerate(results[:3], 1):
            doc = result.get("document", "")
            distance = result.get("distance", 0)
            print(f"    Result {i}: distance={distance:.4f}, text={doc[:80]}...")

        # Verify results come from our test collection
        wrong_collection = [
            r for r in results if r.get("collection") != test_collection
        ]
        if wrong_collection:
            print(f"    ✗ Got results from unexpected collections")
            return False

        # Verify we get reasonable semantic matches (not just random)
        first_doc = results[0].get("document", "").lower() if results else ""
        if "test" not in first_doc and "document" not in first_doc:
            print(f"    ⚠ Top result doesn't mention 'test' or 'document' — may be semantically off")

        print(f"    ✓ Query returned {total} relevant results")
        print()
        return True

    except urllib.error.HTTPError as exc:
        print(f"✗ HTTP error: {exc.code} {exc.reason}")
        body = exc.read().decode("utf-8") if exc.fp else ""
        if body:
            print(f"    Response: {body}")
        return False
    except Exception as exc:
        print(f"✗ Error: {exc}")
        return False


def test_collections(base_url: str) -> bool:
    """Test collection listing and verify test collection is present."""
    print("==> Testing collection listing")

    try:
        response = http_get(f"{base_url}/collections")
        collections = response.get("collections", [])
        print(f"    Found {len(collections)} collections")

        for coll in collections[:5]:
            print(f"      - {coll.get('name')}: {coll.get('count')} items ({coll.get('visibility')})")

        # Verify our test collection exists
        test_coll = next(
            (c for c in collections if c.get("name") == "_integration_test"),
            None,
        )
        if not test_coll:
            print("    ✗ Integration test collection not found")
            return False

        print()
        return True

    except urllib.error.HTTPError as exc:
        print(f"✗ HTTP error: {exc.code} {exc.reason}")
        return False
    except Exception as exc:
        print(f"✗ Error: {exc}")
        return False


def test_delete_collection(base_url: str) -> bool:
    """Test collection deletion — clean up integration test data."""
    print("==> Testing collection deletion (cleanup)")

    test_collection = "_integration_test"
    try:
        http_delete(f"{base_url}/collections/{test_collection}")
        print(f"    Deleted '{test_collection}'")

        # Verify it's gone
        response = http_get(f"{base_url}/collections")
        collections = response.get("collections", [])
        still_exists = any(
            c.get("name") == test_collection for c in collections
        )
        if still_exists:
            print(f"    ✗ Collection still exists after deletion")
            return False

        print(f"    ✓ Collection deleted and verified absent")
        print()
        return True

    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"    Collection already absent (404)")
            return True
        print(f"✗ HTTP error: {exc.code} {exc.reason}")
        return False
    except Exception as exc:
        print(f"✗ Error: {exc}")
        return False


def main():
    """Main entry point."""
    env = load_env()
    base_url = env.get("KNOWLEDGE_SERVICE_URL", os.environ.get("KNOWLEDGE_SERVICE_URL", "")).rstrip("/")

    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return 0

    if not base_url:
        print("Error: KNOWLEDGE_SERVICE_URL not set (check .env or environment)")
        return 1

    print(f"Knowledge Service Health Check")
    print(f"Endpoint: {base_url}")
    print()

    # Always check health
    health_ok = check_health(base_url)

    if not health_ok:
        print("✗ Health check failed")
        return 1

    # Run tests if requested
    if "--test" in sys.argv or "--all" in sys.argv:
        print("==> Running integration tests\n")

        ingest_ok = test_ingestion(base_url)
        if not ingest_ok:
            print("✗ Ingestion test failed")
            return 1

        query_ok = test_query(base_url)
        if not query_ok:
            print("✗ Query test failed")
            return 1

        collections_ok = test_collections(base_url)
        if not collections_ok:
            print("✗ Collections test failed")
            return 1

        delete_ok = test_delete_collection(base_url)
        if not delete_ok:
            print("✗ Collection deletion test failed")
            return 1

        print("✓ All tests passed")

    else:
        print("✓ Health check passed")
        print("\nRun with --test to execute integration tests")

    return 0


if __name__ == "__main__":
    sys.exit(main())
