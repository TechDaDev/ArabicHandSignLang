from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.models.prediction_record import PredictionRecord
from app.models.prediction_session import PredictionSession
from app.models.user import User
from app.schemas.predict import PredictRequest, PredictResponse
from app.services.predictor import predict_from_landmarks
from app.utils.validators import flatten_landmarks, serialize_landmarks
from app.db.session import get_db


router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictResponse)
def run_prediction(
    payload: PredictRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PredictResponse:
    if payload.session_id is not None:
        session_obj = db.scalar(
            select(PredictionSession).where(
                PredictionSession.id == payload.session_id,
                PredictionSession.user_id == current_user.id,
            )
        )
        if session_obj is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    flattened = flatten_landmarks(payload.landmarks)
    result = predict_from_landmarks(flattened, top_k=settings.top_k_predictions)

    if payload.save_to_history:
        record = PredictionRecord(
            user_id=current_user.id,
            session_id=payload.session_id,
            predicted_label_en=result["predicted_label_en"],
            predicted_label_ar=result["predicted_label_ar"],
            confidence=result["confidence"],
            top_predictions=result["top_predictions"],
            landmarks=serialize_landmarks(payload.landmarks),
        )
        db.add(record)
        db.commit()

    return PredictResponse(**result, session_id=payload.session_id)
