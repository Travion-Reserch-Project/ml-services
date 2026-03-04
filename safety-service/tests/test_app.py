import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from unittest.mock import patch


def _mock_downloader():
    """Mock downloader to avoid network calls in tests"""
    return {}


@patch('utils.model_downloader.download_safety_models', side_effect=_mock_downloader)
def test_health_endpoint(mock_download):
    """Test health endpoint returns 200"""
    from app import app
    client = TestClient(app)
    response = client.get("/api/safety/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


@patch('utils.model_downloader.download_safety_models', side_effect=_mock_downloader)
def test_predict_endpoint_heuristic(mock_download):
    """Test prediction works with heuristic fallback (no model)"""
    from app import app
    client = TestClient(app)
    
    payload = {
        "features": {
            "lat": 7.0,
            "lon": 80.0,
            "area_cluster": 1,
            "is_beach": 0,
            "is_crowded": 1,
            "is_tourist_place": 1,
            "is_transit": 0,
            "hour": 21,
            "day_of_week": 5,
            "is_weekend": 1,
            "police_nearby": 0
        }
    }
    
    response = client.post("/api/safety/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "prediction" in data
    # All 7 risk categories should be present
    assert len(data["prediction"]) >= 7
    # Values should be Low/Medium/High
    for risk_type, level in data["prediction"].items():
        assert level in ["Low", "Medium", "High"]


@patch('utils.model_downloader.download_safety_models', side_effect=_mock_downloader)
def test_batch_predict_endpoint(mock_download):
    """Test batch prediction endpoint"""
    from app import app
    client = TestClient(app)
    
    payload = {
        "features_list": [
            {
                "lat": 6.9,
                "lon": 79.8,
                "area_cluster": 0,
                "is_beach": 1,
                "is_crowded": 1,
                "is_tourist_place": 1,
                "is_transit": 0,
                "hour": 22,
                "day_of_week": 5,
                "is_weekend": 1,
                "police_nearby": 0
            },
            {
                "lat": 7.2,
                "lon": 80.6,
                "area_cluster": 1,
                "is_beach": 0,
                "is_crowded": 0,
                "is_tourist_place": 0,
                "is_transit": 1,
                "hour": 10,
                "day_of_week": 2,
                "is_weekend": 0,
                "police_nearby": 1
            }
        ]
    }
    
    response = client.post("/api/safety/batch_predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
