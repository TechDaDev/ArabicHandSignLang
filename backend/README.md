# Arabic Hand Sign Language Backend API — Phase 1

This phase sets up the FastAPI foundation for the Arabic Hand Sign Language mobile backend. It includes configuration management, database session wiring, versioned routing under `/api/v1`, OpenAPI docs, and health endpoints.

## What is included in Phase 1

- FastAPI application entry point
- environment-based settings via `.env`
- SQLAlchemy 2.0 engine and session factory
- PostgreSQL-ready connection configuration
- versioned API router at `/api/v1`
- `GET /api/v1/health`
- `GET /api/v1/health/db`
- interactive API docs at `/docs`

## Project Structure

```text
backend/
├── app/
│   ├── api/
│   ├── core/
│   ├── db/
│   ├── models/
│   ├── schemas/
│   ├── services/
│   ├── utils/
│   └── main.py
├── .env.example
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

### 2. Install packages

```powershell
pip install -r requirements.txt
```

### 3. Create your environment file

```powershell
Copy-Item .env.example .env
```

Update the PostgreSQL values in `.env` to match your local database.

### 4. Run the server

```powershell
uvicorn app.main:app --reload
```

## API Docs

Once the server is running, open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/openapi.json`

## Health Endpoints

- `GET http://127.0.0.1:8000/api/v1/health`
- `GET http://127.0.0.1:8000/api/v1/health/db`

Example response:

```json
{
  "status": "ok"
}
```

Database response when connected:

```json
{
  "status": "ok",
  "database": "connected"
}
```

## Phase 2 Preview

Phase 2 can build on this foundation with domain models, authentication, and prediction endpoints.
