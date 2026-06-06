import pytest
from fastapi.testclient import TestClient
from app.main import app
import pickle
import numpy as np
import os
from sklearn.linear_model import LogisticRegression

client = TestClient(app)


# ── Helper: create a dummy .pkl model for testing ───────────────────────────
def create_dummy_model(path="models/test_model.pkl"):
    os.makedirs("models", exist_ok=True)
    # Train a tiny logistic regression on dummy data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression()
    model.fit(X, y)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


# ── Tests ────────────────────────────────────────────────────────────────────

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"]


def test_list_models_empty():
    response = client.get("/models/")
    assert response.status_code == 200


def test_register_model():
    path = create_dummy_model()
    response = client.post("/models/register", json={
        "model_name": "test-model",
        "version": "v1",
        "file_path": path
    })
    assert response.status_code == 200
    assert "registered" in response.json()["message"]


def test_list_models_after_register():
    response = client.get("/models/")
    assert response.status_code == 200
    models = response.json().get("models", [])
    assert any(m["model_name"] == "test-model" for m in models)


def test_activate_model():
    response = client.put("/models/activate", json={
        "model_name": "test-model",
        "version": "v1"
    })
    assert response.status_code == 200
    assert "active" in response.json()["message"]


def test_activate_nonexistent_model():
    response = client.put("/models/activate", json={
        "model_name": "ghost-model",
        "version": "v99"
    })
    assert response.status_code == 404