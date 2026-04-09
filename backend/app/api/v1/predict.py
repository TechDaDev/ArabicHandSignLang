from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_active_user
from app.models.user import User
from app.schemas.predict import PredictFrameRequest, PredictFrameResponse
from app.services.predictor import PredictionValidationError, predict_frame


router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/frame", response_model=PredictFrameResponse)
def predict_single_frame(
    payload: PredictFrameRequest,
    current_user: User = Depends(get_current_active_user),
) -> PredictFrameResponse:
    """Run authenticated single-frame one-hand inference from 21 landmarks."""
    _ = current_user
    try:
        result = predict_frame(payload.landmarks, top_k=payload.top_k)
    except PredictionValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {exc}") from exc

    return PredictFrameResponse(**result)
