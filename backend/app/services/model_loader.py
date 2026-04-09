from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from app.core.config import settings
from app.core.exceptions import ModelArtifactsUnavailable


@dataclass(frozen=True)
class ModelArtifacts:
    model: Any
    scaler: Any
    label_encoder: Any


@lru_cache(maxsize=1)
def get_model_artifacts() -> ModelArtifacts:
    paths = [Path(settings.model_path), Path(settings.scaler_path), Path(settings.label_encoder_path)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise ModelArtifactsUnavailable(f"Missing ML artifacts: {', '.join(missing)}")

    try:
        return ModelArtifacts(
            model=joblib.load(settings.model_path),
            scaler=joblib.load(settings.scaler_path),
            label_encoder=joblib.load(settings.label_encoder_path),
        )
    except Exception as exc:
        raise ModelArtifactsUnavailable(f"Failed to load ML artifacts: {exc}") from exc
