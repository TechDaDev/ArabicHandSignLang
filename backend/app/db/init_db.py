from app.db.base import Base
from app.db.session import engine


def init_db() -> None:
    """Create database tables for future models when they are added."""
    Base.metadata.create_all(bind=engine)
