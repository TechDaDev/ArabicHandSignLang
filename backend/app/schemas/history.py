from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.predict import LandmarkPoint, TopPrediction


class PredictionRecordSummary(BaseModel):
    id: UUID
    predicted_label: str
    arabic_label: str
    confidence: float
    top_predictions: list[TopPrediction]
    client_timestamp: datetime | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PredictionRecordDetail(PredictionRecordSummary):
    raw_landmarks_json: list[LandmarkPoint] | None = None


class SavedPhraseCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    content: str = Field(min_length=1, max_length=5000)
    source_session_id: UUID | None = None


class SavedPhraseUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    content: str | None = Field(default=None, min_length=1, max_length=5000)


class SavedPhraseResponse(BaseModel):
    id: UUID
    title: str | None = None
    content: str
    source_session_id: UUID | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
