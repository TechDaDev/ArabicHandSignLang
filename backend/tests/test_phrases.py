from fastapi.testclient import TestClient


def test_saved_phrase_crud_and_foreign_access_rejection(client: TestClient, auth_user_factory) -> None:
    user_a = auth_user_factory("phrase_a")
    user_b = auth_user_factory("phrase_b")

    create_response = client.post(
        "/api/v1/history/phrases",
        json={"title": "Greeting", "content": "مرحبا"},
        headers=user_a["headers"],
    )
    assert create_response.status_code == 201
    phrase_id = create_response.json()["id"]

    list_response = client.get("/api/v1/history/phrases", headers=user_a["headers"])
    assert list_response.status_code == 200
    assert len(list_response.json()) == 1

    detail_response = client.get(f"/api/v1/history/phrases/{phrase_id}", headers=user_a["headers"])
    assert detail_response.status_code == 200
    assert detail_response.json()["content"] == "مرحبا"

    update_response = client.patch(
        f"/api/v1/history/phrases/{phrase_id}",
        json={"title": "Greeting Updated", "content": "اهلا"},
        headers=user_a["headers"],
    )
    assert update_response.status_code == 200
    assert update_response.json()["title"] == "Greeting Updated"

    foreign_response = client.get(f"/api/v1/history/phrases/{phrase_id}", headers=user_b["headers"])
    assert foreign_response.status_code == 404

    delete_response = client.delete(f"/api/v1/history/phrases/{phrase_id}", headers=user_a["headers"])
    assert delete_response.status_code == 204
