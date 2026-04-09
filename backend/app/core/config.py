from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    APP_NAME: str = "Arabic Hand Sign Language Backend API"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True
    SECRET_KEY: str = "change-me-for-local-development"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "arabic_hand_sign_db"
    DATABASE_URL: str | None = None

    MODEL_DIR: str = str(REPO_ROOT / "models")
    MODEL_PATH: str = str(REPO_ROOT / "models" / "hand_sign_model.pkl")
    SCALER_PATH: str = str(REPO_ROOT / "models" / "scaler.pkl")
    LABEL_ENCODER_PATH: str = str(REPO_ROOT / "models" / "label_encoder.pkl")

    @property
    def database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
