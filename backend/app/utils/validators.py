from app.core.constants import LANDMARK_COUNT, LANDMARK_VECTOR_SIZE
from app.core.exceptions import InvalidLandmarkPayload
from app.schemas.predict import LandmarkPoint


def flatten_landmarks(landmarks: list[LandmarkPoint]) -> list[float]:
    if len(landmarks) != LANDMARK_COUNT:
        raise InvalidLandmarkPayload(f"Expected {LANDMARK_COUNT} landmarks, received {len(landmarks)}")

    flattened: list[float] = []
    for index, landmark in enumerate(landmarks):
        try:
            coords = [float(landmark.x), float(landmark.y), float(landmark.z)]
        except (TypeError, ValueError) as exc:
            raise InvalidLandmarkPayload(f"Landmark at index {index} must contain numeric x, y, z values") from exc
        flattened.extend(coords)

    if len(flattened) != LANDMARK_VECTOR_SIZE:
        raise InvalidLandmarkPayload(
            f"Expected flattened landmark vector size {LANDMARK_VECTOR_SIZE}, received {len(flattened)}"
        )

    return flattened


def serialize_landmarks(landmarks: list[LandmarkPoint]) -> list[dict[str, float]]:
    return [{"x": float(point.x), "y": float(point.y), "z": float(point.z)} for point in landmarks]
