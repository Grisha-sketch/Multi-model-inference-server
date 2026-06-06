import pytest
from fastapi.testclient import TestClient
from app.main import app
import pickle
import numpy as np
import os
from sklearn.linear_model import LogisticRegression

client = TestClient(app)


def create_and_register_model():
    """Helper to create, save and register a test model."""
    os.makedirs("models", exist_ok=True)
    path = "models/predict_test_model.pkl"

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression()
    model.fit(X, y)

    with open(path, "wb") as f:
        pickle.dump(model, f)

    client.post("/models/register", json={
        "model_name": "predict-model",
        "version": "v1",
        "file_path": path
    })
    return path


def test_predict_valid_input():
    create_and_register_model()
    response = client.post("/predict/", json={
        "model_name": "predict-model",
        "version": "v1",
        "input_data": [[3.0, 4.0]]   # valid 2-feature input
    })
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["model_name"] == "predict-model"


def test_predict_invalid_model():
    response = client.post("/predict/", json={
        "model_name": "does-not-exist",
        "version": "v1",
        "input_data": [[1.0, 2.0]]
    })
    assert response.status_code == 404


def test_metrics_endpoint():
    response = client.get("/predict/metrics")
    assert response.status_code == 200
    assert "metrics" in response.json()