from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import Select, select

from app.api.deps import DbSession, get_current_active_user
from app.models.prediction_record import PredictionRecord
from app.models.user import User
from app.schemas.history import PredictionRecordDetail, PredictionRecordSummary


router = APIRouter(prefix="/history", tags=["history"])


@router.get("/predictions", response_model=list[PredictionRecordSummary])
def list_prediction_history(
    db: DbSession,
    current_user: User = Depends(get_current_active_user),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    predicted_label: str | None = Query(default=None),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
) -> list[PredictionRecordSummary]:
    """Return the authenticated user's stored prediction history."""
    query: Select[tuple[PredictionRecord]] = select(PredictionRecord).where(PredictionRecord.user_id == current_user.id)

    if predicted_label:
        query = query.where(PredictionRecord.predicted_label == predicted_label)
    if min_confidence is not None:
        query = query.where(PredictionRecord.confidence >= min_confidence)

    query = query.order_by(PredictionRecord.created_at.desc()).offset(skip).limit(limit)
    records = db.scalars(query).all()

    return [
        PredictionRecordSummary(
            id=record.id,
            predicted_label=record.predicted_label,
            arabic_label=record.arabic_label,
            confidence=record.confidence,
            top_predictions=record.top_predictions_json,
            client_timestamp=record.client_timestamp,
            created_at=record.created_at,
        )
        for record in records
    ]


@router.get("/predictions/{record_id}", response_model=PredictionRecordDetail)
def get_prediction_record(
    record_id: UUID,
    db: DbSession,
    current_user: User = Depends(get_current_active_user),
) -> PredictionRecordDetail:
    """Return a single stored prediction record belonging to the authenticated user."""
    record = db.scalar(
        select(PredictionRecord).where(
            PredictionRecord.id == record_id,
            PredictionRecord.user_id == current_user.id,
        )
    )
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction record not found")

    return PredictionRecordDetail(
        id=record.id,
        predicted_label=record.predicted_label,
        arabic_label=record.arabic_label,
        confidence=record.confidence,
        top_predictions=record.top_predictions_json,
        client_timestamp=record.client_timestamp,
        created_at=record.created_at,
        raw_landmarks_json=record.raw_landmarks_json,
    )
