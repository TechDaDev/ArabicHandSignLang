# PROJECT_MASTER_CONTEXT.md

## Project Name
Arabic Hand Sign Language Backend API

## Project Type
FastAPI backend for a Flutter mobile application

## Project Goal
Build a clean, production-oriented FastAPI backend for an Arabic Hand Sign Language mobile system.

The Flutter mobile app will run MediaPipe on-device and extract hand landmarks locally.  
The backend will receive lightweight landmark vectors, run inference using the trained ML artifacts from the existing ArabicHandSignLang project, and return predictions, confidence, top predictions, session data, and user history.

This is an API-first project.

---

## Core Product Direction

### Mobile App Role
The Flutter app will:
- authenticate users
- run MediaPipe locally on-device
- extract one-hand landmark vectors
- send landmarks to the FastAPI backend
- receive predicted Arabic letters and confidence
- optionally group letters into words locally
- optionally save recognized phrases and history to backend

### Backend Role
The FastAPI backend will:
- manage authentication
- manage users
- load trained ML model artifacts
- validate incoming landmarks
- run inference
- store prediction history
- store prediction sessions
- store saved phrases
- store user feedback

---

## Important Constraints

Do NOT add any of the following to the project plan or implementation:
- Redis
- Celery
- background jobs
- WebSockets
- Docker
- Kubernetes
- microservices
- Django
- admin dashboard
- cloud deployment files for now

Keep the architecture:
- simple
- clean
- modular
- mobile-oriented
- scalable later without overengineering now

---

## Tech Stack

Use:
- FastAPI
- Python 3.11+
- SQLAlchemy 2.0
- Pydantic v2
- Alembic
- PostgreSQL
- JWT authentication
- passlib / bcrypt or pwdlib for password hashing
- joblib for model loading
- scikit-learn inference using saved artifacts

Use pytest for tests.

---

## Existing ML Artifacts To Integrate

The backend must integrate these trained model artifacts from the existing repo:

- `models/hand_sign_model.pkl`
- `models/scaler.pkl`
- `models/label_encoder.pkl`

The backend must NOT retrain the model.

The backend must load the model artifacts once and reuse them in memory.

---

## ML Inference Contract

### Current Input Assumption
For v1, support **one hand only**.

Each request contains:
- 21 landmarks
- each landmark has x, y, z
- total flattened features = 63

The current saved model expects **63 features**.

### Input Validation Rules
Reject requests if:
- number of landmarks is not 21
- any landmark is missing x/y/z
- any x/y/z is not numeric
- final flattened vector length is not 63

### Inference Output
The API should return:
- predicted English label
- Arabic label
- confidence
- top predictions
- timestamp

---

## Arabic Label Mapping

Use the same label mapping logic as in the existing project.

Expected mapping includes labels like:
- Ain -> ع
- Alef -> أ or ا depending on final project consistency
- Beh -> ب
- Jeem -> ج
- Kaf -> ك
- Lam -> ل
- Meem -> م
- Noon -> ن
- Qaf -> ق
- Reh -> ر
- Seen -> س
- Sheen -> ش
- Teh -> ت
- Theh -> ث
- Waw -> و
- Yeh -> ي
- Zain -> ز
- etc.

This mapping should be centralized in one service or constants file.

Do not duplicate it across routes.

---

## Project Structure

Build the backend using this structure:

```text
backend/
├── app/
│   ├── api/
│   │   ├── deps.py
│   │   ├── router.py
│   │   └── v1/
│   │       ├── auth.py
│   │       ├── users.py
│   │       ├── predict.py
│   │       ├── sessions.py
│   │       ├── history.py
│   │       └── feedback.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   ├── constants.py
│   │   └── exceptions.py
│   ├── db/
│   │   ├── base.py
│   │   ├── session.py
│   │   └── init_db.py
│   ├── models/
│   │   ├── user.py
│   │   ├── prediction_session.py
│   │   ├── prediction_record.py
│   │   ├── saved_phrase.py
│   │   └── feedback.py
│   ├── schemas/
│   │   ├── auth.py
│   │   ├── user.py
│   │   ├── predict.py
│   │   ├── session.py
│   │   ├── history.py
│   │   └── feedback.py
│   ├── services/
│   │   ├── model_loader.py
│   │   ├── predictor.py
│   │   ├── label_mapper.py
│   │   └── session_builder.py
│   ├── utils/
│   │   ├── timestamps.py
│   │   └── validators.py
│   └── main.py
├── alembic/
├── tests/
├── requirements.txt
├── .env.example
└── README.md