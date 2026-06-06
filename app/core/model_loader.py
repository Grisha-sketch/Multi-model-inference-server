import pickle
import joblib
import onnxruntime as ort


def load_model(file_path: str):
    """
    Load a model from disk based on its file extension.
    Supports .pkl, .joblib, and .onnx formats.
    Returns the loaded model object.
    """

    if file_path.endswith(".pkl"):
        # Load scikit-learn or any pickle-serialized model
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        print(f"[ModelLoader] Loaded .pkl model from {file_path}")
        return model

    elif file_path.endswith(".joblib"):
        # Load joblib-serialized model (common for scikit-learn)
        model = joblib.load(file_path)
        print(f"[ModelLoader] Loaded .joblib model from {file_path}")
        return model

    elif file_path.endswith(".onnx"):
        # Load ONNX model using onnxruntime InferenceSession
        model = ort.InferenceSession(file_path)
        print(f"[ModelLoader] Loaded .onnx model from {file_path}")
        return model

    else:
        raise ValueError(f"Unsupported model format: {file_path}. Use .pkl, .joblib, or .onnx")