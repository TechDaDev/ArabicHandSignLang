from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import Select, select

from app.api.deps import DbSession, get_current_active_user
from app.models.prediction_record import PredictionRecord
from app.models.prediction_session import PredictionSession
from app.models.saved_phrase import SavedPhrase
from app.models.user import User
from app.schemas.history import (
    PredictionRecordDetail,
    PredictionRecordSummary,
    SavedPhraseCreateRequest,
    SavedPhraseResponse,
    SavedPhraseUpdateRequest,
)


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


@router.post("/phrases", response_model=SavedPhraseResponse, status_code=status.HTTP_201_CREATED)
def create_saved_phrase(
    payload: SavedPhraseCreateRequest,
    db: DbSession,
    current_user: User = Depends(get_current_active_user),
) -> SavedPhrase:
    """Create a saved phrase for the authenticated user."""
    if payload.source_session_id is not None:
        session_obj = db.scalar(
            select(PredictionSession).where(
                PredictionSession.id == payload.source_session_id,
                PredictionSession.user_id == current_user.id,
            )
        )
        if session_obj is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    phrase = SavedPhrase(
        user_id=current_user.id,
        title=payload.title,
        content=payload.content,
        source_session_id=payload.source_session_id,
    )
    db.add(phrase)
    db.commit()
    db.refresh(phrase)
    return phrase


@router.get("/phrases", response_model=list[SavedPhraseResponse])
def list_saved_phrases(
    db: DbSession,
    current_user: User = Depends(get_current_active_user),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[SavedPhrase]:
    """List the authenticated user's saved phrases, newest first."""
    return db.scalars(
        select(SavedPhrase)
        .where(SavedPhrase.user_id == current_user.id)
        .order_by(SavedPhrase.created_at.desc())
        .offset(skip)
        .limit(limit)
    ).all()


@router.get("/phrases/{phrase_id}", response_model=SavedPhraseResponse)
def get_saved_phrase(
    phrase_id: UUID,
    db: DbSession,
    current_user: User = Depends(get_current_active_user),
) -> SavedPhrase:
    """Return one saved phrase belonging to the authenticated user."""
    phrase = db.scalar(
        select(SavedPhrase).where(
            SavedPhrase.id == phrase_id,
            SavedPhrase.user_id == current_user.id,
        )
    )
    if phrase is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Saved phrase not found")
    return phrase


@router.patch("/phrases/{phrase_id}", response_model=SavedPhraseResponse)
def update_saved_phrase(
    phrase_id: UUID,
    payload: SavedPhraseUpdateRequest,
    db: DbSession,
    current_user: User = Depends(get_current_active_user),
) -> SavedPhrase:
    """Update a saved phrase owned by the authenticated user."""
    phrase = db.scalar(
        select(SavedPhrase).where(
            SavedPhrase.id == phrase_id,
            SavedPhrase.user_id == current_user.id,
        )
    )
    if phrase is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Saved phrase not found")

    updates = payload.model_dump(exclude_unset=True)
    for field_name in ("title", "content"):
        if field_name in updates:
            setattr(phrase, field_name, updates[field_name])

    db.add(phrase)
    db.commit()
    db.refresh(phrase)
    return phrase


@router.delete("/phrases/{phrase_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_saved_phrase(
    phrase_id: UUID,
    db: DbSession,
    current_user: User = Depends(get_current_active_user),
) -> Response:
    """Delete a saved phrase owned by the authenticated user."""
    phrase = db.scalar(
        select(SavedPhrase).where(
            SavedPhrase.id == phrase_id,
            SavedPhrase.user_id == current_user.id,
        )
    )
    if phrase is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Saved phrase not found")

    db.delete(phrase)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
