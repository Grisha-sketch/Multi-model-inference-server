import os
from fastapi import FastAPI
from app.routers import models, predict
from app.middleware.logging_middleware import LoggingMiddleware

# Auto-create required folders
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Create FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(
    title="Multi-Model Inference Server",
    description="Upload, register, and serve multiple ML models through a single REST API.",
    version="1.0.0"
)

# ── Attach middleware ────────────────────────────────────────────────────────
# LoggingMiddleware runs automatically on every request
app.add_middleware(LoggingMiddleware)

# ── Register routers ─────────────────────────────────────────────────────────
app.include_router(models.router)   # /models endpoints
app.include_router(predict.router)  # /predict endpoints


# ── Root endpoint ────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Multi-Model Inference Server is running.",
        "docs": "Visit /docs for the interactive API interface."
    }