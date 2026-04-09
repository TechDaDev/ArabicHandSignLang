from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.history import PredictionRecordSummary


class SessionStartRequest(BaseModel):
    notes: str | None = Field(default=None, max_length=2000)


class SessionSummary(BaseModel):
    id: UUID
    status: str
    notes: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    prediction_count: int


class SessionDetail(SessionSummary):
    recent_predictions: list[PredictionRecordSummary] = []
