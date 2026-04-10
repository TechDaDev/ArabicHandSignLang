from pathlib import Path

from app.core.config import BASE_DIR, Settings


def test_default_model_paths_use_backend_directory() -> None:
    settings = Settings()

    assert Path(settings.MODEL_DIR) == BASE_DIR / "models"
    assert Path(settings.MODEL_PATH) == BASE_DIR / "models" / "hand_sign_model.pkl"
    assert Path(settings.SCALER_PATH) == BASE_DIR / "models" / "scaler.pkl"
    assert Path(settings.LABEL_ENCODER_PATH) == BASE_DIR / "models" / "label_encoder.pkl"
