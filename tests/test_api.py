import sys
from pathlib import Path

# Supports both `pytest` from the project root and VS Code's "Run Python File".
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()
def test_invalid_batch():
    assert client.post("/predict/batch", json=[]).status_code == 422
