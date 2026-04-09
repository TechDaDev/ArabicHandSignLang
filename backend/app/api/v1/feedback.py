from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.feedback import Feedback
from app.models.user import User
from app.schemas.feedback import FeedbackCreate, FeedbackResponse


router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def create_feedback(
    payload: FeedbackCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Feedback:
    feedback_item = Feedback(
        user_id=current_user.id,
        session_id=payload.session_id,
        feedback_type=payload.feedback_type,
        message=payload.message,
        is_correct=payload.is_correct,
        expected_label_en=payload.expected_label_en,
    )
    db.add(feedback_item)
    db.commit()
    db.refresh(feedback_item)
    return feedback_item


@router.get("", response_model=list[FeedbackResponse])
def list_feedback(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Feedback]:
    return db.scalars(
        select(Feedback).where(Feedback.user_id == current_user.id).order_by(Feedback.created_at.desc())
    ).all()
