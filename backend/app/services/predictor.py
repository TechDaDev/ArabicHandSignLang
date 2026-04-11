import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.core.config import settings
from app.core.constants import FEATURE_NAMES
from app.schemas.predict import LandmarkPoint
from app.services.label_mapper import get_arabic_label
from app.services.model_loader import load_model_artifacts


logger = logging.getLogger(__name__)
SCANNING_LABEL = "Scanning..."
SCANNING_ARABIC_LABEL = "جاري الفحص..."


class PredictionValidationError(ValueError):
    """Raised when the landmark payload cannot be transformed for inference."""


def _flatten_landmarks(landmarks: list[LandmarkPoint]) -> list[float]:
    if len(landmarks) != 21:
        raise PredictionValidationError("Expected exactly 21 landmarks for one-hand prediction")

    flattened: list[float] = []
    for point in landmarks:
        flattened.extend([float(point.x), float(point.y), float(point.z)])

    if len(flattened) != 63:
        raise PredictionValidationError("Expected 63 flattened landmark features")

    return flattened


def _decode_label(raw_label: Any, label_encoder: Any) -> str:
    if isinstance(raw_label, str):
        return raw_label
    try:
        return str(label_encoder.inverse_transform([raw_label])[0])
    except Exception:
        return str(raw_label)


def predict_frame(landmarks: list[LandmarkPoint], top_k: int = 3) -> dict[str, Any]:
    """Run one-frame landmark inference and return a mobile-friendly response."""
    artifacts = load_model_artifacts()
    flattened = _flatten_landmarks(landmarks)

    features = pd.DataFrame([flattened], columns=FEATURE_NAMES)
    scaled_features = artifacts.scaler.transform(features)

    top_predictions: list[dict[str, Any]] = []
    predicted_label: str
    arabic_label: str
    confidence: float

    if hasattr(artifacts.model, "predict_proba"):
        probabilities = artifacts.model.predict_proba(scaled_features)[0]
        classes = getattr(artifacts.model, "classes_", range(len(probabilities)))
        for model_class, probability in zip(classes, probabilities):
            label = _decode_label(model_class, artifacts.label_encoder)
            top_predictions.append(
                {
                    "label": label,
                    "arabic_label": get_arabic_label(label),
                    "confidence": round(float(probability), 6),
                }
            )
        top_predictions.sort(key=lambda item: item["confidence"], reverse=True)
        top_predictions = top_predictions[:top_k]

        top_entry = top_predictions[0]
        predicted_label = top_entry["label"]
        arabic_label = top_entry["arabic_label"]
        confidence = top_entry["confidence"]
    else:
        predicted_raw = artifacts.model.predict(scaled_features)[0]
        predicted_label = _decode_label(predicted_raw, artifacts.label_encoder)
        arabic_label = get_arabic_label(predicted_label)
        confidence = 1.0
        top_predictions = [
            {
                "label": predicted_label,
                "arabic_label": arabic_label,
                "confidence": confidence,
            }
        ]

    confidence_threshold = float(settings.PREDICTION_CONFIDENCE_THRESHOLD)
    is_confident = confidence >= confidence_threshold

    logger.info(
        "Prediction label=%s confidence=%.6f threshold=%.2f is_confident=%s",
        predicted_label,
        confidence,
        confidence_threshold,
        is_confident,
    )

    return {
        "predicted_label": predicted_label if is_confident else SCANNING_LABEL,
        "arabic_label": arabic_label if is_confident else SCANNING_ARABIC_LABEL,
        "confidence": confidence,
        "top_predictions": top_predictions,
        "timestamp": datetime.now(timezone.utc),
        "is_confident": is_confident,
        "confidence_threshold": confidence_threshold,
    }
