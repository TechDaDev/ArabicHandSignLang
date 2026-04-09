from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

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
