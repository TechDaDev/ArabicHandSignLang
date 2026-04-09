from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.prediction_record import PredictionRecord
from app.models.saved_phrase import SavedPhrase
from app.models.user import User
from app.schemas.history import PredictionRecordResponse, SavedPhraseCreate, SavedPhraseResponse


router = APIRouter(prefix="/history", tags=["history"])


@router.get("/predictions", response_model=list[PredictionRecordResponse])
def list_prediction_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[PredictionRecord]:
    return db.scalars(
        select(PredictionRecord)
        .where(PredictionRecord.user_id == current_user.id)
        .order_by(PredictionRecord.created_at.desc())
    ).all()


@router.get("/phrases", response_model=list[SavedPhraseResponse])
def list_saved_phrases(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[SavedPhrase]:
    return db.scalars(
        select(SavedPhrase)
        .where(SavedPhrase.user_id == current_user.id)
        .order_by(SavedPhrase.created_at.desc())
    ).all()


@router.post("/phrases", response_model=SavedPhraseResponse, status_code=status.HTTP_201_CREATED)
def save_phrase(
    payload: SavedPhraseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SavedPhrase:
    phrase = SavedPhrase(
        user_id=current_user.id,
        session_id=payload.session_id,
        phrase=payload.phrase,
        language=payload.language,
    )
    db.add(phrase)
    db.commit()
    db.refresh(phrase)
    return phrase
