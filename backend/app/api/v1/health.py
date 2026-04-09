from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.deps import DbSession


router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    status: str


class DatabaseHealthResponse(HealthResponse):
    database: str


@router.get("", response_model=HealthResponse)
def read_health() -> HealthResponse:
    """Return a simple service status response."""
    return HealthResponse(status="ok")


@router.get("/db", response_model=DatabaseHealthResponse)
def read_database_health(db: DbSession) -> DatabaseHealthResponse:
    """Run a lightweight database connectivity check."""
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "error", "database": "disconnected", "message": str(exc)},
        ) from exc

    return DatabaseHealthResponse(status="ok", database="connected")
