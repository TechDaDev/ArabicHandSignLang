from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import DbSession, get_current_active_user
from app.models.prediction_record import PredictionRecord
from app.models.user import User
from app.schemas.predict import PredictFrameRequest, PredictFrameResponse
from app.services.predictor import PredictionValidationError, predict_frame


router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/frame", response_model=PredictFrameResponse)
def predict_single_frame(
    payload: PredictFrameRequest,
    db: DbSession,
    current_user: User = Depends(get_current_active_user),
) -> PredictFrameResponse:
    """Run authenticated single-frame one-hand inference from 21 landmarks."""
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
