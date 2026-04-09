# Arabic Hand Sign Language Backend API

A modular `FastAPI` backend for the Arabic Hand Sign Language mobile app. The backend now supports authentication, one-frame landmark inference using the trained ML artifacts, prediction history, sessions, feedback, and saved phrases.

## Implemented Features

- JWT auth and current-user profile endpoints
- single-frame one-hand prediction from 21 landmarks / 63 features
- prediction history storage and retrieval
- prediction session start / predict / end lifecycle
- feedback submission linked to prediction records or sessions
- saved phrase CRUD
- OpenAPI docs and health endpoints
- Alembic migration readiness and pytest coverage

## Project Structure

```text
backend/
├── alembic/
├── app/
│   ├── api/
│   ├── core/
│   ├── db/
│   ├── models/
│   ├── schemas/
│   ├── services/
│   └── main.py
├── tests/
├── .env.example
├── alembic.ini
├── README.md
└── requirements.txt
```

## Setup

### 1. Create and activate a virtual environment

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Create your environment file

```powershell
Copy-Item .env.example .env
```

Update `.env` for your local PostgreSQL database, or temporarily use SQLite for quick local testing:

```env
DATABASE_URL=sqlite:///./app.db
```

### 4. Run the API server

```powershell
uvicorn app.main:app --reload
```

## API Docs

Open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/openapi.json`

## Test Commands

Run the full minimal backend suite:

```powershell
pytest tests -q
```

## Migration Commands

From `backend/`:

```powershell
alembic revision --autogenerate -m "create initial backend schema"
alembic upgrade head
```

## Implemented Endpoint Summary

### Health
- `GET /api/v1/health`
- `GET /api/v1/health/db`

### Auth and User
- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`
- `GET /api/v1/users/me`
- `PATCH /api/v1/users/me`

### Prediction
- `POST /api/v1/predict/frame`

### Prediction History
- `GET /api/v1/history/predictions`
- `GET /api/v1/history/predictions/{id}`

### Sessions
- `POST /api/v1/sessions/start`
- `POST /api/v1/sessions/{session_id}/predict-frame`
- `POST /api/v1/sessions/{session_id}/end`
- `GET /api/v1/sessions`
- `GET /api/v1/sessions/{session_id}`

### Feedback
- `POST /api/v1/feedback`
- `GET /api/v1/feedback/me`

### Saved Phrases
- `POST /api/v1/history/phrases`
- `GET /api/v1/history/phrases`
- `GET /api/v1/history/phrases/{id}`
- `PATCH /api/v1/history/phrases/{id}`
- `DELETE /api/v1/history/phrases/{id}`

## Basic Usage Flow

1. Register a user with `POST /api/v1/auth/register`
2. Log in with `POST /api/v1/auth/login`
3. Use the bearer token for protected endpoints
4. Send one-frame predictions to `POST /api/v1/predict/frame`
5. Review stored history in `GET /api/v1/history/predictions`
6. Optionally start and manage a prediction session
7. Save feedback or phrases as needed
