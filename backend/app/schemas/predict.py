from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float


class PredictFrameRequest(BaseModel):
    landmarks: list[LandmarkPoint]
    top_k: int = Field(default=3, ge=1, le=5)
    client_timestamp: datetime | None = None

    @field_validator("landmarks")
    @classmethod
    def validate_landmark_count(cls, value: list[LandmarkPoint]) -> list[LandmarkPoint]:
        if len(value) != 21:
            raise ValueError("Exactly 21 landmarks are required")
        return value


class TopPrediction(BaseModel):
    label: str
    arabic_label: str
    confidence: float


class PredictFrameResponse(BaseModel):
    predicted_label: str
    arabic_label: str
    confidence: float
    top_predictions: list[TopPrediction]
    timestamp: datetime
