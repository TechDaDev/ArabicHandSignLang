from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[Optional[int]] = mapped_column(ForeignKey("prediction_sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    predicted_label_en: Mapped[str] = mapped_column(String(100), index=True)
    predicted_label_ar: Mapped[str] = mapped_column(String(100))
    confidence: Mapped[float] = mapped_column(Float)
    top_predictions: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    landmarks: Mapped[list[dict[str, float]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    user = relationship("User", back_populates="prediction_records")
    session = relationship("PredictionSession", back_populates="records")
