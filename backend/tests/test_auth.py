from fastapi.testclient import TestClient


def test_register_success_and_duplicate_email_rejected(client: TestClient) -> None:
    payload = {
        "email": "auth@example.com",
        "password": "Secret123!",
        "username": "auth_user",
        "full_name": "Auth User",
    }

    first = client.post("/api/v1/auth/register", json=payload)
    assert first.status_code == 201
    assert first.json()["email"] == payload["email"]

    duplicate = client.post("/api/v1/auth/register", json=payload)
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "Email already exists"


def test_login_success_and_invalid_password_rejected(client: TestClient) -> None:
    payload = {
        "email": "login@example.com",
        "password": "Secret123!",
        "username": "login_user",
    }
    client.post("/api/v1/auth/register", json=payload)

    success = client.post(
        "/api/v1/auth/login",
        json={"email": payload["email"], "password": payload["password"]},
    )
    assert success.status_code == 200
    assert "access_token" in success.json()

    invalid = client.post(
        "/api/v1/auth/login",
        json={"email": payload["email"], "password": "WrongPass123!"},
    )
    assert invalid.status_code == 401
    assert invalid.json()["detail"] == "Invalid email or password"


def test_auth_me_requires_authentication(client: TestClient) -> None:
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401
