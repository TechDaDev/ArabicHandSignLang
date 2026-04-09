from fastapi.testclient import TestClient


def test_feedback_creation_list_and_foreign_record_rejection(
    client: TestClient,
    auth_user_factory,
    sample_landmarks: list[dict[str, float]],
) -> None:
    user_a = auth_user_factory("feedback_a")
    user_b = auth_user_factory("feedback_b")

    predict_response = client.post(
        "/api/v1/predict/frame",
        json={"landmarks": sample_landmarks, "top_k": 3},
        headers=user_a["headers"],
    )
    assert predict_response.status_code == 200

    history_response = client.get("/api/v1/history/predictions", headers=user_a["headers"])
    record_id = history_response.json()[0]["id"]

    create_response = client.post(
        "/api/v1/feedback",
        json={"prediction_record_id": record_id, "is_correct": True, "expected_label": "Noon", "notes": "Looks good"},
        headers=user_a["headers"],
    )
    assert create_response.status_code == 201

    list_response = client.get("/api/v1/feedback/me", headers=user_a["headers"])
    assert list_response.status_code == 200
    assert len(list_response.json()) == 1

    foreign_response = client.post(
        "/api/v1/feedback",
        json={"prediction_record_id": record_id, "is_correct": False},
        headers=user_b["headers"],
    )
    assert foreign_response.status_code == 404
