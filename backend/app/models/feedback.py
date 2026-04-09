from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[Optional[int]] = mapped_column(ForeignKey("prediction_sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    feedback_type: Mapped[str] = mapped_column(String(50), default="general")
    message: Mapped[str] = mapped_column(Text)
    is_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    expected_label_en: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="feedback_entries")
    session = relationship("PredictionSession", back_populates="feedback_entries")
