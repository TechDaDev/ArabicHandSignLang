from fastapi.testclient import TestClient


def test_prediction_is_stored_and_listed_for_owner_only(
    client: TestClient,
    auth_user_factory,
    sample_landmarks: list[dict[str, float]],
) -> None:
    user_a = auth_user_factory("history_a")
    user_b = auth_user_factory("history_b")

    predict_response = client.post(
        "/api/v1/predict/frame",
        json={"landmarks": sample_landmarks, "top_k": 3},
        headers=user_a["headers"],
    )
    assert predict_response.status_code == 200

    list_response = client.get("/api/v1/history/predictions", headers=user_a["headers"])
    assert list_response.status_code == 200
    assert len(list_response.json()) == 1

    record_id = list_response.json()[0]["id"]
    detail_response = client.get(f"/api/v1/history/predictions/{record_id}", headers=user_a["headers"])
    assert detail_response.status_code == 200

    foreign_response = client.get(f"/api/v1/history/predictions/{record_id}", headers=user_b["headers"])
    assert foreign_response.status_code == 404
