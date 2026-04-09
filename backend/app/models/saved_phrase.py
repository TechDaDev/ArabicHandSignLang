from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class SavedPhrase(Base):
    __tablename__ = "saved_phrases"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[Optional[int]] = mapped_column(ForeignKey("prediction_sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    phrase: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(10), default="ar")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="saved_phrases")
    session = relationship("PredictionSession", back_populates="saved_phrases")
