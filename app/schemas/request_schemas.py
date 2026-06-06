from pydantic import BaseModel
from typing import Any, List, Optional


class RegisterModelRequest(BaseModel):
    model_name: str          # e.g. "iris-classifier"
    version: str             # e.g. "v1"
    file_path: str           # path where the uploaded file was saved


class ActivateModelRequest(BaseModel):
    model_name: str
    version: str


class PredictRequest(BaseModel):
    model_name: str          # which model to use
    version: str             # which version
    input_data: List[Any]    # raw input, e.g. [[5.1, 3.5, 1.4, 0.2]]


class PredictResponse(BaseModel):
    model_name: str
    version: str
    prediction: Any
    confidence: Optional[float] = None   # not all models return confidence


class ModelInfo(BaseModel):
    model_name: str
    version: str
    file_path: str
    is_active: bool