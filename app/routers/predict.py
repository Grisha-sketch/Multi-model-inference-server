import time
from fastapi import APIRouter, HTTPException
from app.core.model_registry import registry
from app.core.predictor import run_prediction
from app.metrics.tracker import tracker
from app.schemas.request_schemas import PredictRequest, PredictResponse

router = APIRouter(prefix="/predict", tags=["Predict"])


@router.post("/", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Run inference using a registered model.
    Provide model_name, version, and input_data in the request body.
    """
    start = time.time()

    try:
        # Get the requested model from the registry
        model = registry.get_model(
            model_name=request.model_name,
            version=request.version
        )

        # Run prediction
        prediction, confidence = run_prediction(model, request.input_data)

        # Record latency metric
        latency_ms = round((time.time() - start) * 1000, 2)
        tracker.record_request(request.model_name, request.version, latency_ms)

        return PredictResponse(
            model_name=request.model_name,
            version=request.version,
            prediction=prediction,
            confidence=confidence
        )

    except ValueError as e:
        tracker.record_error(request.model_name, request.version)
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        tracker.record_error(request.model_name, request.version)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/metrics")
def get_metrics():
    """
    Return performance metrics for all models.
    Shows request count, average latency, and error count.
    """
    return {"metrics": tracker.get_metrics()}