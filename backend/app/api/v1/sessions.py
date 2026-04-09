from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select

from app.api.deps import DbSession, get_current_active_user
from app.models.prediction_record import PredictionRecord
from app.models.prediction_session import PredictionSession
from app.models.user import User
from app.schemas.history import PredictionRecordSummary
from app.schemas.predict import PredictFrameRequest, PredictFrameResponse
from app.schemas.session import SessionDetail, SessionStartRequest, SessionSummary
from app.services.predictor import PredictionValidationError, predict_frame


router = APIRouter(prefix="/sessions", tags=["sessions"])


def _get_owned_session(db: DbSession, session_id: UUID, user_id: UUID) -> PredictionSession:
    session_obj = db.scalar(
        select(PredictionSession).where(
            PredictionSession.id == session_id,
            PredictionSession.user_id == user_id,
        )
    )
    if session_obj is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return session_obj


def _build_session_summary(db: DbSession, session_obj: PredictionSession) -> SessionSummary:
    prediction_count = db.scalar(
        select(func.count()).select_from(PredictionRecord).where(PredictionRecord.session_id == session_obj.id)
    ) or 0
    return SessionSummary(
        id=session_obj.id,
        status=session_obj.status,
        notes=session_obj.notes,
        started_at=session_obj.started_at,
        ended_at=session_obj.ended_at,
        prediction_count=int(prediction_count),
    )


@router.post("/start", response_model=SessionSummary, status_code=status.HTTP_201_CREATED)
def start_session(
    payload: SessionStartRequest,
    db: DbSession,
    current_user: User = Depends(get_current_active_user),
) -> SessionSummary:
    """Create a new active prediction session for the authenticated user."""
    session_obj = PredictionSession(user_id=current_user.id, status="active", notes=payload.notes)
    db.add(session_obj)
    db.commit()
    db.refresh(session_obj)
    return _build_session_summary(db, session_obj)


@router.post("/{session_id}/predict-frame", response_model=PredictFrameResponse)
def predict_in_session(
    session_id: UUID,
    payload: PredictFrameRequest,
    db: DbSession,
    current_user: User = Depends(get_current_active_user),
) -> PredictFrameResponse:
    """Run a frame prediction and link the stored record to an active session."""
    session_obj = _get_owned_session(db, session_id, current_user.id)
    if session_obj.status != "active":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Session is not active")

    try:
        result = predict_frame(payload.landmarks, top_k=payload.top_k)
    except PredictionValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {exc}") from exc

    prediction_record = PredictionRecord(
        user_id=current_user.id,
        session_id=session_obj.id,
        predicted_label=result["predicted_label"],
        arabic_label=result["arabic_label"],
        confidence=result["confidence"],
        top_predictions_json=result["top_predictions"],
        raw_landmarks_json=[point.model_dump() for point in payload.landmarks],
        client_timestamp=payload.client_timestamp,
    )
    db.add(prediction_record)
    db.commit()

    return PredictFrameResponse(**result)


@router.post("/{session_id}/end", response_model=SessionSummary)
def end_session(
    session_id: UUID,
    db: DbSession,
    current_user: User = Depends(get_current_active_user),
) -> SessionSummary:
    """Mark an active session as completed."""
    session_obj = _get_owned_session(db, session_id, current_user.id)
    if session_obj.status != "active":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Session is already closed")

    session_obj.status = "completed"
    session_obj.ended_at = datetime.now(timezone.utc)
    db.add(session_obj)
    db.commit()
    db.refresh(session_obj)
    return _build_session_summary(db, session_obj)


@router.get("", response_model=list[SessionSummary])
def list_sessions(
    db: DbSession,
    current_user: User = Depends(get_current_active_user),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[SessionSummary]:
    """List the authenticated user's prediction sessions, newest first."""
    sessions = db.scalars(
        select(PredictionSession)
        .where(PredictionSession.user_id == current_user.id)
        .order_by(PredictionSession.started_at.desc())
        .offset(skip)
        .limit(limit)
    ).all()
    return [_build_session_summary(db, session_obj) for session_obj in sessions]


@router.get("/{session_id}", response_model=SessionDetail)
def get_session_detail(
    session_id: UUID,
    db: DbSession,
    current_user: User = Depends(get_current_active_user),
) -> SessionDetail:
    """Return one owned session with a compact list of recent prediction records."""
    session_obj = _get_owned_session(db, session_id, current_user.id)
    prediction_records = db.scalars(
        select(PredictionRecord)
        .where(
            PredictionRecord.user_id == current_user.id,
            PredictionRecord.session_id == session_obj.id,
        )
        .order_by(PredictionRecord.created_at.desc())
        .limit(10)
    ).all()

    summary = _build_session_summary(db, session_obj)
    recent_predictions = [
        PredictionRecordSummary(
            id=record.id,
            predicted_label=record.predicted_label,
            arabic_label=record.arabic_label,
            confidence=record.confidence,
            top_predictions=record.top_predictions_json,
            client_timestamp=record.client_timestamp,
            created_at=record.created_at,
        )
        for record in prediction_records
    ]

    return SessionDetail(**summary.model_dump(), recent_predictions=recent_predictions)
