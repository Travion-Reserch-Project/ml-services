# Weather Service Tests
# Basic API structure validation (no ML models required during tests)

import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200


def test_api_documentation(client):
    """Test that OpenAPI documentation is available"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data


def test_invalid_endpoint(client):
    """Test that invalid endpoints return 404"""
    response = client.get("/api/invalid")
    assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
