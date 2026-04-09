import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class PredictionRecord(Base):
    """Stored authenticated frame prediction record."""

    __tablename__ = "prediction_records"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    predicted_label: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    arabic_label: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    top_predictions_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    raw_landmarks_json: Mapped[list[dict[str, float]] | None] = mapped_column(JSON, nullable=True)
    client_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
