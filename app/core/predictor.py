import numpy as np
import onnxruntime as ort


def run_prediction(model, input_data: list):
    """
    Run inference on a loaded model.
    Automatically detects model type and handles accordingly.
    Returns prediction and confidence (if available).
    """

    # Convert input to numpy array
    data = np.array(input_data, dtype=np.float32)

    # ── ONNX model ──────────────────────────────────────────
    if isinstance(model, ort.InferenceSession):
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: data})
        prediction = outputs[0].tolist()
        confidence = None  # ONNX models may not always return confidence
        return prediction, confidence

    # ── Scikit-learn model (.pkl / .joblib) ─────────────────
    else:
        prediction = model.predict(data).tolist()

        # Try to get confidence/probability if model supports it
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(data)
            confidence = float(np.max(proba))  # highest class probability

        return prediction, confidence