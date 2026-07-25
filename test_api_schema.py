"""Integration test to validate API response schema."""
from fastapi.testclient import TestClient

from api.app import app

REQUIRED_KEYS = {
    "url",
    "name",
    "adaptive_support",
    "description",
    "duration",
    "remote_support",
    "test_type",
}


def test_recommend_schema():
    client = TestClient(app)
    resp = client.post("/recommend", json={"query": "Data analyst with Python and SQL"})
    assert resp.status_code == 200, f"API returned {resp.status_code}: {resp.text}" 
    data = resp.json()
    assert "recommended_assessments" in data, "Missing recommended_assessments key"
    for item in data["recommended_assessments"]:
        assert set(item.keys()) == REQUIRED_KEYS, f"Schema mismatch: {item.keys()}"
