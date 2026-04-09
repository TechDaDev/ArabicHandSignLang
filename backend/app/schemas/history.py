from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionRecordResponse(BaseModel):
    id: int
    session_id: int | None = None
    predicted_label_en: str
    predicted_label_ar: str
    confidence: float
    top_predictions: list[dict[str, Any]]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SavedPhraseCreate(BaseModel):
    phrase: str = Field(min_length=1, max_length=1000)
    session_id: int | None = None
    language: str = "ar"


class SavedPhraseResponse(BaseModel):
    id: int
    session_id: int | None = None
    phrase: str
    language: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
