import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.model_registry import registry
from app.schemas.request_schemas import RegisterModelRequest, ActivateModelRequest

router = APIRouter(prefix="/models", tags=["Models"])

UPLOAD_DIR = "models"  # folder where uploaded files are saved
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_model(file: UploadFile = File(...)):
    """
    Upload a model file (.pkl, .joblib, .onnx) to the server.
    The file is saved to the /models folder.
    """
    allowed_extensions = (".pkl", ".joblib", ".onnx")

    if not file.filename.endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )

    save_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "Model file uploaded successfully.",
        "file_path": save_path
    }


@router.post("/register")
def register_model(request: RegisterModelRequest):
    """
    Register an uploaded model file with a name and version.
    Automatically loads it into memory.
    """
    try:
        registry.register(
            model_name=request.model_name,
            version=request.version,
            file_path=request.file_path
        )
        return {
            "message": f"Model '{request.model_name}:{request.version}' registered successfully."
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.get("/")
def list_models():
    """
    List all registered models with their version and active status.
    """
    models = registry.list_models()
    if not models:
        return {"message": "No models registered yet.", "models": []}
    return {"models": models}


@router.put("/activate")
def activate_model(request: ActivateModelRequest):
    """
    Switch the active version of a model without restarting the server.
    """
    try:
        registry.activate(
            model_name=request.model_name,
            version=request.version
        )
        return {
            "message": f"'{request.model_name}:{request.version}' is now the active model."
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))