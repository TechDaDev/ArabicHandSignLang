from fastapi.testclient import TestClient


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
