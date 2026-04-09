from fastapi.testclient import TestClient


def sample_landmarks() -> list[dict[str, float]]:
    return [
        {"x": round(index * 0.01, 4), "y": round(index * 0.02, 4), "z": round(index * -0.01, 4)}
        for index in range(21)
    ]


def test_register_login_and_profile(client: TestClient) -> None:
    register_payload = {
        "email": "mobile.user@example.com",
        "username": "mobile_user",
        "password": "Secret123",
    }
    register_response = client.post("/api/v1/auth/register", json=register_payload)
    assert register_response.status_code == 201
    assert register_response.json()["email"] == register_payload["email"]

    login_response = client.post(
        "/api/v1/auth/login",
        json={"email": register_payload["email"], "password": register_payload["password"]},
    )
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]

    profile_response = client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {token}"})
    assert profile_response.status_code == 200
    assert profile_response.json()["username"] == register_payload["username"]


def test_prediction_history_and_feedback_flow(client: TestClient, auth_headers: dict[str, str]) -> None:
    session_response = client.post(
        "/api/v1/sessions",
        json={"title": "Demo session", "notes": "Smoke test"},
        headers=auth_headers,
    )
    assert session_response.status_code == 201
    session_id = session_response.json()["id"]

    prediction_response = client.post(
        "/api/v1/predict",
        json={"landmarks": sample_landmarks(), "session_id": session_id, "save_to_history": True},
        headers=auth_headers,
    )
    assert prediction_response.status_code == 200
    prediction_body = prediction_response.json()
    assert prediction_body["predicted_label_en"]
    assert prediction_body["predicted_label_ar"]
    assert 0.0 <= prediction_body["confidence"] <= 1.0
    assert len(prediction_body["top_predictions"]) >= 1

    history_response = client.get("/api/v1/history/predictions", headers=auth_headers)
    assert history_response.status_code == 200
    assert len(history_response.json()) == 1

    phrase_response = client.post(
        f"/api/v1/sessions/{session_id}/phrases",
        json={"phrase": "مرحبا"},
        headers=auth_headers,
    )
    assert phrase_response.status_code == 201

    feedback_response = client.post(
        "/api/v1/feedback",
        json={"session_id": session_id, "message": "Prediction looks good", "is_correct": True},
        headers=auth_headers,
    )
    assert feedback_response.status_code == 201

    session_detail_response = client.get(f"/api/v1/sessions/{session_id}", headers=auth_headers)
    assert session_detail_response.status_code == 200
    assert session_detail_response.json()["summary"]["total_predictions"] == 1


def test_predict_rejects_invalid_landmark_count(client: TestClient, auth_headers: dict[str, str]) -> None:
    invalid_landmarks = sample_landmarks()[:-1]
    response = client.post(
        "/api/v1/predict",
        json={"landmarks": invalid_landmarks, "save_to_history": False},
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert "Expected 21 landmarks" in response.json()["detail"]
