from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Arabic Hand Sign Language API"
    app_version: str = "0.1.0"
    debug: bool = False

    api_v1_prefix: str = "/api/v1"
    secret_key: str = "change-this-secret-before-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    database_url: str = f"sqlite:///{(BACKEND_DIR / 'app.db').as_posix()}"
    postgres_database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/arabic_sign_api"

    model_path: str = str(REPO_ROOT / "models" / "hand_sign_model.pkl")
    scaler_path: str = str(REPO_ROOT / "models" / "scaler.pkl")
    label_encoder_path: str = str(REPO_ROOT / "models" / "label_encoder.pkl")
    top_k_predictions: int = 3

    cors_origins: list[str] = ["*"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
