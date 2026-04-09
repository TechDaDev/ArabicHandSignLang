from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FeedbackCreate(BaseModel):
    session_id: int | None = None
    feedback_type: str = "general"
    message: str = Field(min_length=1, max_length=2000)
    is_correct: bool | None = None
    expected_label_en: str | None = None


class FeedbackResponse(BaseModel):
    id: int
    session_id: int | None = None
    feedback_type: str
    message: str
    is_correct: bool | None = None
    expected_label_en: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
