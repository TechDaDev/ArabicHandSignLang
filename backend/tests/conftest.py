import os
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

TEST_DB_PATH = Path(__file__).resolve().parent / "test_suite.db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH.as_posix()}"

from app.db.base import Base
from app.db.session import engine
from app.main import app


@pytest.fixture(autouse=True)
def reset_database() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def sample_landmarks() -> list[dict[str, float]]:
    return [
        {"x": round(index * 0.01, 4), "y": round(index * 0.02, 4), "z": round(index * -0.01, 4)}
        for index in range(21)
    ]


@pytest.fixture()
def auth_user_factory(client: TestClient):
    def _create_user(prefix: str = "user", password: str = "Secret123!") -> dict[str, str]:
        suffix = uuid4().hex[:8]
        email = f"{prefix}_{suffix}@example.com"
        username = f"{prefix}_{suffix}"

        register_response = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password, "username": username, "full_name": prefix.title()},
        )
        assert register_response.status_code == 201

        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": password},
        )
        assert login_response.status_code == 200

        token = login_response.json()["access_token"]
        return {
            "email": email,
            "username": username,
            "password": password,
            "headers": {"Authorization": f"Bearer {token}"},
        }

    return _create_user
