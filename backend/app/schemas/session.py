from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.history import PredictionRecordSummary
from app.schemas.predict import PredictFrameResponse


class SessionStartRequest(BaseModel):
    notes: str | None = Field(default=None, max_length=2000)


class SessionSummary(BaseModel):
    id: UUID
    status: str
    notes: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    prediction_count: int


class SessionPredictFrameResponse(PredictFrameResponse):
    stable_label: str | None = None
    stable_arabic_label: str | None = None
    is_stable: bool = False
    stable_count: int = 0
    current_word: str = ""
    text_buffer: str = ""
    session_status: str


class SessionDetail(SessionSummary):
    recent_predictions: list[PredictionRecordSummary] = []
    recent_raw_predictions_window: list[str] = []
    current_word: str = ""
    text_buffer: str = ""
    stable_label: str | None = None
    stable_arabic_label: str | None = None
    is_stable: bool = False
    stable_count: int = 0
    last_stable_label: str | None = None
    last_stable_arabic_label: str | None = None
    last_committed_label: str | None = None
    last_committed_arabic_label: str | None = None
    last_commit_timestamp: datetime | None = None
