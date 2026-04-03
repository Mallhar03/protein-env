"""
tests/unit/test_app_logic_audit.py — Coverage booster for Section 3 audit.

Tests the FastAPI entry points in server/app.py using TestClient, 
ensuring the HTTP-to-Environment wiring is functional.
"""

from fastapi.testclient import TestClient
import pytest
from unittest.mock import patch

from server.app import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """GET /health returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_metadata_endpoint(client):
    """GET /metadata returns environment info."""
    response = client.get("/metadata")
    assert response.status_code == 200
    assert "name" in response.json()


def test_schema_endpoint(client):
    """GET /schema returns Pydantic models."""
    response = client.get("/schema")
    assert response.status_code == 200
    assert "action" in response.json()


@patch("server.app._env")
def test_reset_endpoint(mock_env, client):
    """POST /reset calls environment reset."""
    mock_env.reset.return_value = {
        "protein_id": "test",
        "sequence": "MA",
        "task_type": "easy",
        "task_description": "test",
        "available_tools": [],
        "step_number": 0,
        "max_steps": 10
    }
    
    response = client.post(
        "/reset", 
        json={"task_type": "easy", "seed": 123}
    )
    assert response.status_code == 200
    assert response.json()["protein_id"] == "test"
    mock_env.reset.assert_called_once()
