from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.prediction_session import PredictionSession
from app.models.saved_phrase import SavedPhrase
from app.models.user import User
from app.schemas.history import SavedPhraseCreate, SavedPhraseResponse
from app.schemas.session import (
    PredictionSessionCreate,
    PredictionSessionResponse,
    PredictionSessionUpdate,
    SessionSummary,
)
from app.services.session_builder import build_session_summary


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=PredictionSessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    payload: PredictionSessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PredictionSessionResponse:
    session_obj = PredictionSession(user_id=current_user.id, title=payload.title, notes=payload.notes)
    db.add(session_obj)
    db.commit()
    db.refresh(session_obj)
    return PredictionSessionResponse.model_validate(session_obj, from_attributes=True)


@router.get("", response_model=list[PredictionSessionResponse])
def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[PredictionSessionResponse]:
    sessions = db.scalars(
        select(PredictionSession)
        .where(PredictionSession.user_id == current_user.id)
        .order_by(PredictionSession.started_at.desc())
    ).all()
    return [PredictionSessionResponse.model_validate(item, from_attributes=True) for item in sessions]


@router.get("/{session_id}", response_model=PredictionSessionResponse)
def get_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PredictionSessionResponse:
    session_obj = db.scalar(
        select(PredictionSession)
        .options(selectinload(PredictionSession.records), selectinload(PredictionSession.saved_phrases))
        .where(PredictionSession.id == session_id, PredictionSession.user_id == current_user.id)
    )
    if session_obj is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    response = PredictionSessionResponse.model_validate(session_obj, from_attributes=True)
    response.summary = SessionSummary(**build_session_summary(session_obj.records, session_obj.saved_phrases))
    return response


@router.patch("/{session_id}", response_model=PredictionSessionResponse)
def update_session(
    session_id: int,
    payload: PredictionSessionUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PredictionSessionResponse:
    session_obj = db.scalar(
        select(PredictionSession).where(PredictionSession.id == session_id, PredictionSession.user_id == current_user.id)
    )
    if session_obj is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if payload.title is not None:
        session_obj.title = payload.title
    if payload.notes is not None:
        session_obj.notes = payload.notes
    if payload.ended_at is not None:
        session_obj.ended_at = payload.ended_at

    db.add(session_obj)
    db.commit()
    db.refresh(session_obj)
    return PredictionSessionResponse.model_validate(session_obj, from_attributes=True)


@router.post("/{session_id}/phrases", response_model=SavedPhraseResponse, status_code=status.HTTP_201_CREATED)
def save_phrase_to_session(
    session_id: int,
    payload: SavedPhraseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SavedPhrase:
    session_obj = db.scalar(
        select(PredictionSession).where(PredictionSession.id == session_id, PredictionSession.user_id == current_user.id)
    )
    if session_obj is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    saved_phrase = SavedPhrase(
        user_id=current_user.id,
        session_id=session_id,
        phrase=payload.phrase,
        language=payload.language,
    )
    db.add(saved_phrase)
    db.commit()
    db.refresh(saved_phrase)
    return saved_phrase
