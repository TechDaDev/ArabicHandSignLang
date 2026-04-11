from fastapi.testclient import TestClient

import app.api.v1.sessions as sessions_module


def test_session_lifecycle_and_foreign_access_are_enforced(
    client: TestClient,
    auth_user_factory,
    sample_landmarks: list[dict[str, float]],
) -> None:
    user_a = auth_user_factory("session_a")
    user_b = auth_user_factory("session_b")

    start_response = client.post(
        "/api/v1/sessions/start",
        json={"notes": "Testing session"},
        headers=user_a["headers"],
    )
    assert start_response.status_code == 201
    session_id = start_response.json()["id"]

    predict_response = client.post(
        f"/api/v1/sessions/{session_id}/predict-frame",
        json={"landmarks": sample_landmarks, "top_k": 3},
        headers=user_a["headers"],
    )
    assert predict_response.status_code == 200

    end_response = client.post(f"/api/v1/sessions/{session_id}/end", headers=user_a["headers"])
    assert end_response.status_code == 200
    assert end_response.json()["status"] == "completed"

    closed_response = client.post(
        f"/api/v1/sessions/{session_id}/predict-frame",
        json={"landmarks": sample_landmarks, "top_k": 3},
        headers=user_a["headers"],
    )
    assert closed_response.status_code == 409

    foreign_response = client.get(f"/api/v1/sessions/{session_id}", headers=user_b["headers"])
    assert foreign_response.status_code == 404


def test_session_prediction_exposes_stabilized_state_and_prevents_repeat_commits(
    client: TestClient,
    auth_user_factory,
    sample_landmarks: list[dict[str, float]],
    monkeypatch,
) -> None:
    user = auth_user_factory("session_stable")

    start_response = client.post(
        "/api/v1/sessions/start",
        json={"notes": "Stabilized session"},
        headers=user["headers"],
    )
    assert start_response.status_code == 201
    session_id = start_response.json()["id"]

    monkeypatch.setattr(
        sessions_module,
        "predict_frame",
        lambda landmarks, top_k=3: {
            "predicted_label": "Alef",
            "arabic_label": "أ",
            "confidence": 0.95,
            "top_predictions": [
                {"label": "Alef", "arabic_label": "أ", "confidence": 0.95},
                {"label": "Beh", "arabic_label": "ب", "confidence": 0.03},
            ],
            "timestamp": "2026-04-11T10:00:00Z",
            "is_confident": True,
            "confidence_threshold": 0.45,
        },
    )

    early_response = None
    for _ in range(4):
        early_response = client.post(
            f"/api/v1/sessions/{session_id}/predict-frame",
            json={"landmarks": sample_landmarks, "top_k": 3},
            headers=user["headers"],
        )
        assert early_response.status_code == 200

    early_body = early_response.json()
    assert early_body["is_stable"] is False
    assert early_body["current_word"] == ""
    assert early_body["text_buffer"] == ""

    stabilized_response = None
    for _ in range(8):
        stabilized_response = client.post(
            f"/api/v1/sessions/{session_id}/predict-frame",
            json={"landmarks": sample_landmarks, "top_k": 3},
            headers=user["headers"],
        )
        assert stabilized_response.status_code == 200

    body = stabilized_response.json()
    assert body["stable_label"] == "Alef"
    assert body["stable_arabic_label"] == "أ"
    assert body["is_stable"] is True
    assert body["current_word"] == "أ"
    assert body["text_buffer"] == "أ"
    assert body["session_status"] == "active"

    repeat_response = client.post(
        f"/api/v1/sessions/{session_id}/predict-frame",
        json={"landmarks": sample_landmarks, "top_k": 3},
        headers=user["headers"],
    )
    assert repeat_response.status_code == 200
    repeat_body = repeat_response.json()
    assert repeat_body["text_buffer"] == "أ"

    detail_response = client.get(f"/api/v1/sessions/{session_id}", headers=user["headers"])
    assert detail_response.status_code == 200
    detail_body = detail_response.json()
    assert detail_body["current_word"] == "أ"
    assert detail_body["text_buffer"] == "أ"
    assert detail_body["last_stable_label"] == "Alef"
    assert detail_body["last_committed_label"] == "Alef"
