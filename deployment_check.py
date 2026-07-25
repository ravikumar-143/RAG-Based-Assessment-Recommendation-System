"""Deployment checklist validator."""
from __future__ import annotations
import os
from typing import Dict, Any

import requests

from embeddings.index_builder import load_index
from reranker.rerank import _client  # type: ignore
from utils import validate_recommendation_schema


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def check_health() -> bool:
    resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
    return resp.status_code == 200 and resp.json().get("status") == "healthy"


def check_recommend_schema() -> bool:
    payload = {"query": "Data analyst with Python and SQL"}
    resp = requests.post(f"{API_BASE_URL}/recommend", json=payload, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"/recommend failed with status {resp.status_code}: {resp.text}")
    data: Dict[str, Any] = resp.json()
    if "recommended_assessments" not in data:
        raise RuntimeError("Missing recommended_assessments in response")
    validate_recommendation_schema(data["recommended_assessments"])
    return True


def check_faiss_loads() -> bool:
    load_index()
    return True


def check_groq_active() -> bool:
    _client()
    return True


def main() -> None:
    results = {}
    checks = {
        "api_reachable": check_health,
        "api_schema": check_recommend_schema,
        "faiss_load": check_faiss_loads,
        "groq_active": check_groq_active,
    }

    for name, fn in checks.items():
        try:
            ok = fn()
            results[name] = ok
        except Exception as exc:  # noqa: BLE001
            results[name] = False
            print(f"[FAIL] {name}: {exc}")
        else:
            print(f"[OK] {name}")

    all_passed = all(results.values())
    if not all_passed:
        raise SystemExit("Deployment checklist failed; see logs above.")
    print("All deployment checks passed.")


if __name__ == "__main__":
    main()
