# Multi-Model Inference Server

A FastAPI-based REST API that serves multiple ML models through a single server.
Supports `.pkl`, `.joblib`, and `.onnx` model formats with hot-swapping and metrics.

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

## Run Locally

```bash
uvicorn app.main:app --reload
```

Visit: http://localhost:8000/docs

## Run with Docker

```bash
docker compose up --build
```

## API Endpoints

| Method | Endpoint           | Description                  |
|--------|--------------------|------------------------------|
| POST   | /models/upload     | Upload a model file          |
| POST   | /models/register   | Register model with name+ver |
| GET    | /models/           | List all registered models   |
| PUT    | /models/activate   | Hot-swap active model        |
| POST   | /predict/          | Run inference                |
| GET    | /predict/metrics   | View performance metrics     |

## Run Tests

```bash
pytest tests/
``` 