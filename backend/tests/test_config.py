from app.core.config import Settings


def test_database_url_normalizes_postgres_driver_scheme() -> None:
    settings = Settings(DATABASE_URL="postgresql://user:pass@localhost:5432/appdb")

    assert settings.database_url == "postgresql+psycopg://user:pass@localhost:5432/appdb"


def test_database_url_normalizes_legacy_postgres_scheme() -> None:
    settings = Settings(DATABASE_URL="postgres://user:pass@localhost:5432/appdb")

    assert settings.database_url == "postgresql+psycopg://user:pass@localhost:5432/appdb"
