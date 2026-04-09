from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PredictionSessionCreate(BaseModel):
    title: str | None = None
    notes: str | None = None


class PredictionSessionUpdate(BaseModel):
    title: str | None = None
    notes: str | None = None
    ended_at: datetime | None = None


class SessionSummary(BaseModel):
    total_predictions: int
    average_confidence: float | None = None
    latest_phrase: str | None = None


class PredictionSessionResponse(BaseModel):
    id: int
    user_id: int
    title: str | None = None
    notes: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    summary: SessionSummary | None = None

    model_config = ConfigDict(from_attributes=True)
