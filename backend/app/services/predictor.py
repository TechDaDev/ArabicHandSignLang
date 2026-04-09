from typing import Any

import pandas as pd

from app.core.constants import FEATURE_NAMES
from app.services.label_mapper import get_arabic_label
from app.services.model_loader import get_model_artifacts
from app.utils.timestamps import utc_now


def _decode_label(raw_label: Any, label_encoder: Any) -> str:
    if isinstance(raw_label, str):
        return raw_label
    try:
        return str(label_encoder.inverse_transform([raw_label])[0])
    except Exception:
        return str(raw_label)


def predict_from_landmarks(flat_landmarks: list[float], top_k: int = 3) -> dict[str, Any]:
    artifacts = get_model_artifacts()
    features = pd.DataFrame([flat_landmarks], columns=FEATURE_NAMES)
    features_scaled = artifacts.scaler.transform(features)

    predicted_raw = artifacts.model.predict(features_scaled)[0]
    predicted_label_en = _decode_label(predicted_raw, artifacts.label_encoder)

    top_predictions: list[dict[str, Any]] = []
    if hasattr(artifacts.model, "predict_proba"):
        probabilities = artifacts.model.predict_proba(features_scaled)[0]
        model_classes = getattr(artifacts.model, "classes_", [])
        for model_class, probability in zip(model_classes, probabilities):
            english_label = _decode_label(model_class, artifacts.label_encoder)
            top_predictions.append(
                {
                    "english_label": english_label,
                    "arabic_label": get_arabic_label(english_label),
                    "confidence": float(probability),
                }
            )
        top_predictions.sort(key=lambda item: item["confidence"], reverse=True)
        top_predictions = top_predictions[:top_k]
    else:
        top_predictions = [
            {
                "english_label": predicted_label_en,
                "arabic_label": get_arabic_label(predicted_label_en),
                "confidence": 1.0,
            }
        ]

    return {
        "predicted_label_en": predicted_label_en,
        "predicted_label_ar": get_arabic_label(predicted_label_en),
        "confidence": float(top_predictions[0]["confidence"]),
        "top_predictions": top_predictions,
        "timestamp": utc_now(),
    }
