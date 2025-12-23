import pytest
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.app import app, load_artifacts

@pytest.fixture
def client():
    # Load artifacts before testing
    load_artifacts()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"

def test_predict_valid(client):
    payload = {
        "ticket_id": "test-123",
        "subject": "Login failed",
        "description": "I cannot login to my account even with correct password."
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["ticket_id"] == "test-123"
    assert "predicted_category" in data
    assert "confidence" in data
    # "Login failed" is typically a Bug Report or Technical Issue
    assert isinstance(data["predicted_category"], str)

def test_predict_missing_fields(client):
    payload = {
        "ticket_id": "test-124"
        # Missing subject and description
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 400

def test_predict_invalid_json(client):
    response = client.post('/predict', data="not json")
    assert response.status_code == 400
