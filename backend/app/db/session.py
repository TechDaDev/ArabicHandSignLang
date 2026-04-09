from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings


engine_options: dict[str, Any] = {"pool_pre_ping": True}
if settings.database_url.startswith("postgresql"):
    engine_options["connect_args"] = {"connect_timeout": 5}

engine = create_engine(settings.database_url, **engine_options)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session for request-scoped usage."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
