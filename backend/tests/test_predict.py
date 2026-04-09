from fastapi.testclient import TestClient


def test_valid_authenticated_frame_prediction_works(
    client: TestClient,
    auth_user_factory,
    sample_landmarks: list[dict[str, float]],
) -> None:
    user = auth_user_factory("predict")
    response = client.post(
        "/api/v1/predict/frame",
        json={"landmarks": sample_landmarks, "top_k": 3},
        headers=user["headers"],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"]
    assert body["arabic_label"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top_predictions"]) >= 1


def test_invalid_landmark_count_is_rejected(
    client: TestClient,
    auth_user_factory,
    sample_landmarks: list[dict[str, float]],
) -> None:
    user = auth_user_factory("predict_invalid")
    response = client.post(
        "/api/v1/predict/frame",
        json={"landmarks": sample_landmarks[:-1], "top_k": 3},
        headers=user["headers"],
    )

    assert response.status_code == 422


def test_unauthenticated_prediction_is_rejected(
    client: TestClient,
    sample_landmarks: list[dict[str, float]],
) -> None:
    response = client.post(
        "/api/v1/predict/frame",
        json={"landmarks": sample_landmarks, "top_k": 3},
    )
    assert response.status_code == 401
