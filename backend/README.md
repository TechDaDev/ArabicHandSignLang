# Arabic Hand Sign Language Backend API

A modular FastAPI backend for a Flutter mobile application that sends one-hand MediaPipe landmarks to the server for Arabic sign-letter prediction.

## Features

- JWT authentication
- SQLAlchemy 2.0 models for users, sessions, history, saved phrases, and feedback
- Inference using the existing ML artifacts in the repository
- Centralized Arabic label mapping
- Validation for 21 landmarks / 63 features
- Pytest coverage for the core API flow

## Project Layout

```text
backend/
├── app/
├── alembic/
├── tests/
├── .env.example
└── requirements.txt
```

## Quick Start

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Open:
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

## Main API Endpoints

- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `GET /api/v1/users/me`
- `POST /api/v1/predict`
- `GET/POST /api/v1/sessions`
- `GET /api/v1/history/predictions`
- `GET/POST /api/v1/history/phrases`
- `GET/POST /api/v1/feedback`

## Notes

- The backend reuses `models/hand_sign_model.pkl`, `models/scaler.pkl`, and `models/label_encoder.pkl` from the repository root.
- Production should point `DATABASE_URL` to PostgreSQL.
- For tests and quick local runs, SQLite is supported as a simple fallback.
