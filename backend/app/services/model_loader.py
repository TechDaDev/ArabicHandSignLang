from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from app.core.config import settings


@dataclass(frozen=True)
class ModelArtifacts:
    model: Any
    scaler: Any
    label_encoder: Any


@lru_cache(maxsize=1)
def load_model_artifacts() -> ModelArtifacts:
    """Load and cache the trained ML artifacts exactly once."""
    model_path = Path(settings.MODEL_PATH)
    scaler_path = Path(settings.SCALER_PATH)
    label_encoder_path = Path(settings.LABEL_ENCODER_PATH)

    missing_paths = [
        str(path)
        for path in (model_path, scaler_path, label_encoder_path)
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(f"Missing model artifacts: {', '.join(missing_paths)}")

    return ModelArtifacts(
        model=joblib.load(model_path),
        scaler=joblib.load(scaler_path),
        label_encoder=joblib.load(label_encoder_path),
    )
