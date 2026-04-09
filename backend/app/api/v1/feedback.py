from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select

from app.api.deps import DbSession, get_current_active_user
from app.models.feedback import Feedback
from app.models.prediction_record import PredictionRecord
from app.models.prediction_session import PredictionSession
from app.models.user import User
from app.schemas.feedback import FeedbackCreateRequest, FeedbackResponse


router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def create_feedback(
    payload: FeedbackCreateRequest,
    db: DbSession,
    current_user: User = Depends(get_current_active_user),
) -> FeedbackResponse:
    """Store feedback for the authenticated user with strict ownership checks."""
    if payload.prediction_record_id is not None:
        record = db.scalar(
            select(PredictionRecord).where(
                PredictionRecord.id == payload.prediction_record_id,
                PredictionRecord.user_id == current_user.id,
            )
        )
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction record not found")

    if payload.session_id is not None:
        session_obj = db.scalar(
            select(PredictionSession).where(
                PredictionSession.id == payload.session_id,
                PredictionSession.user_id == current_user.id,
            )
        )
        if session_obj is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    feedback_item = Feedback(
        user_id=current_user.id,
        prediction_record_id=payload.prediction_record_id,
        session_id=payload.session_id,
        is_correct=payload.is_correct,
        expected_label=payload.expected_label,
        notes=payload.notes,
    )
    db.add(feedback_item)
    db.commit()
    db.refresh(feedback_item)

    return FeedbackResponse(
        id=feedback_item.id,
        prediction_record_id=feedback_item.prediction_record_id,
        session_id=feedback_item.session_id,
        is_correct=feedback_item.is_correct,
        expected_label=feedback_item.expected_label,
        notes=feedback_item.notes,
        created_at=feedback_item.created_at,
    )


@router.get("/me", response_model=list[FeedbackResponse])
def list_my_feedback(
    db: DbSession,
    current_user: User = Depends(get_current_active_user),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[FeedbackResponse]:
    """List the authenticated user's feedback records, newest first."""
    items = db.scalars(
        select(Feedback)
        .where(Feedback.user_id == current_user.id)
        .order_by(Feedback.created_at.desc())
        .offset(skip)
        .limit(limit)
    ).all()

    return [
        FeedbackResponse(
            id=item.id,
            prediction_record_id=item.prediction_record_id,
            session_id=item.session_id,
            is_correct=item.is_correct,
            expected_label=item.expected_label,
            notes=item.notes,
            created_at=item.created_at,
        )
        for item in items
    ]
