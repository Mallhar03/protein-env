"""
tests/integration/test_server.py
Integration tests for the ProteinEnv FastAPI server.
"""

import pytest
from fastapi.testclient import TestClient
from server.app import app

# Basic TestClient for integration validations
client = TestClient(app)

def test_health_check_endpoint():
    """Verify that the orchestrator health check endpoint is functional."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ok"
    assert data["env"] == "protein-env"
    assert "version" in data
