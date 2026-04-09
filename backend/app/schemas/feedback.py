from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class FeedbackCreateRequest(BaseModel):
    prediction_record_id: UUID | None = None
    session_id: UUID | None = None
    is_correct: bool
    expected_label: str | None = Field(default=None, max_length=100)
    notes: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_target(self) -> "FeedbackCreateRequest":
        if self.prediction_record_id is None and self.session_id is None:
            raise ValueError("Either prediction_record_id or session_id must be provided")
        return self


class FeedbackResponse(BaseModel):
    id: UUID
    prediction_record_id: UUID | None = None
    session_id: UUID | None = None
    is_correct: bool
    expected_label: str | None = None
    notes: str | None = None
    created_at: datetime
