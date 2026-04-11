from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


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
    PREDICTION_CONFIDENCE_THRESHOLD: float = 0.45

    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "arabic_hand_sign_db"
    DATABASE_URL: str | None = None

    MODEL_DIR: str = str(BASE_DIR / "models")
    MODEL_PATH: str = str(BASE_DIR / "models" / "hand_sign_model.pkl")
    SCALER_PATH: str = str(BASE_DIR / "models" / "scaler.pkl")
    LABEL_ENCODER_PATH: str = str(BASE_DIR / "models" / "label_encoder.pkl")

    @staticmethod
    def _normalize_database_url(url: str) -> str:
        if url.startswith("postgresql+"):
            return url
        if url.startswith("postgresql://"):
            return "postgresql+psycopg://" + url[len("postgresql://") :]
        if url.startswith("postgres://"):
            return "postgresql+psycopg://" + url[len("postgres://") :]
        return url

    @property
    def database_url(self) -> str:
        if self.DATABASE_URL:
            return self._normalize_database_url(self.DATABASE_URL)
        return self._normalize_database_url(
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
