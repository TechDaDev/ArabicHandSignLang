from datetime import datetime

from pydantic import BaseModel, Field


class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float


class PredictRequest(BaseModel):
    landmarks: list[LandmarkPoint]
    session_id: int | None = None
    save_to_history: bool = True


class TopPrediction(BaseModel):
    english_label: str
    arabic_label: str
    confidence: float


class PredictResponse(BaseModel):
    predicted_label_en: str
    predicted_label_ar: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_predictions: list[TopPrediction]
    timestamp: datetime
    session_id: int | None = None
