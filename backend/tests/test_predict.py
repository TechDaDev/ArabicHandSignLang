from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.services.predictor as predictor_module


class _IdentityScaler:
    def transform(self, features):
        return features


class _FakeLabelEncoder:
    def inverse_transform(self, values):
        mapping = {0: "Alef", 1: "Beh", 2: "Teh"}
        return [mapping[int(value)] for value in values]


class _ProbabilityFirstModel:
    classes_ = [0, 1, 2]

    def predict(self, _features):
        return [0]

    def predict_proba(self, _features):
        return [[0.2, 0.7, 0.1]]


class _LowConfidenceModel:
    classes_ = [0, 1, 2]

    def predict(self, _features):
        return [1]

    def predict_proba(self, _features):
        return [[0.4, 0.35, 0.25]]


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


def test_primary_label_comes_from_top_probability(
    client: TestClient,
    auth_user_factory,
    sample_landmarks: list[dict[str, float]],
    monkeypatch,
) -> None:
    user = auth_user_factory("predict_probability")
    monkeypatch.setattr(
        predictor_module,
        "load_model_artifacts",
        lambda: SimpleNamespace(
            model=_ProbabilityFirstModel(),
            scaler=_IdentityScaler(),
            label_encoder=_FakeLabelEncoder(),
        ),
    )

    response = client.post(
        "/api/v1/predict/frame",
        json={"landmarks": sample_landmarks, "top_k": 3},
        headers=user["headers"],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] == "Beh"
    assert body["arabic_label"] == "ب"
    assert body["is_confident"] is True
    assert body["top_predictions"][0]["label"] == body["predicted_label"]


def test_low_confidence_prediction_returns_scanning_state(
    client: TestClient,
    auth_user_factory,
    sample_landmarks: list[dict[str, float]],
    monkeypatch,
) -> None:
    user = auth_user_factory("predict_scanning")
    monkeypatch.setattr(
        predictor_module,
        "load_model_artifacts",
        lambda: SimpleNamespace(
            model=_LowConfidenceModel(),
            scaler=_IdentityScaler(),
            label_encoder=_FakeLabelEncoder(),
        ),
    )

    response = client.post(
        "/api/v1/predict/frame",
        json={"landmarks": sample_landmarks, "top_k": 3},
        headers=user["headers"],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] == "Scanning..."
    assert body["arabic_label"] == "جاري الفحص..."
    assert body["is_confident"] is False
    assert body["confidence_threshold"] == 0.45
    assert body["top_predictions"][0]["label"] == "Alef"
