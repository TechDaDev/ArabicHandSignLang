# Arabic Hand Sign Language Backend API

**Frontend target:** Flutter team  
**Base URL:** `https://ahs-production-7427.up.railway.app`  
**API Prefix:** `/api/v1`

> Use the full base URL with `https` in Flutter:
> `https://ahs-production-7427.up.railway.app`

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Authentication](#authentication)
3. [Common Response Notes](#common-response-notes)
4. [Endpoint Summary](#endpoint-summary)
5. [Root & Health](#root--health)
6. [Auth Endpoints](#auth-endpoints)
7. [User Endpoints](#user-endpoints)
8. [Prediction Endpoints](#prediction-endpoints)
9. [History & Saved Phrases](#history--saved-phrases)
10. [Session Endpoints](#session-endpoints)
11. [Feedback Endpoints](#feedback-endpoints)
12. [Flutter Integration Notes](#flutter-integration-notes)

---

## Quick Start

### Required headers

For **public** endpoints:
```http
Content-Type: application/json
```

For **authenticated** endpoints:
```http
Content-Type: application/json
Authorization: Bearer <access_token>
```

### Common data types

- `id`: UUID string
- `created_at`, `updated_at`, `timestamp`, `started_at`, `ended_at`: ISO-8601 datetime string
- `confidence`: float between `0` and `1`

---

## Authentication

### Login flow for Flutter

1. Register using `POST /api/v1/auth/register` if needed.
2. Login using `POST /api/v1/auth/login`.
3. Save `access_token` securely.
4. Send it in all protected requests:
   ```http
   Authorization: Bearer <token>
   ```

---

## Common Response Notes

### Success
- `200 OK` → request succeeded
- `201 Created` → item created successfully
- `204 No Content` → item deleted successfully

### Common errors
- `401 Unauthorized` → token missing or invalid
- `403 Forbidden` → inactive user or blocked action
- `404 Not Found` → requested item does not exist or is not owned by the user
- `409 Conflict` → duplicate email/username or invalid session state
- `422 Unprocessable Entity` → invalid request body or validation failure
- `500 Internal Server Error` → server/model/runtime issue

Typical error body:
```json
{
  "detail": "Error message here"
}
```

---

## Endpoint Summary

| Method | Endpoint | Auth Required | Purpose |
|---|---|---:|---|
| GET | `/` | No | Root info |
| GET | `/api/v1/health` | No | Service health |
| GET | `/api/v1/health/db` | No | Database health |
| POST | `/api/v1/auth/register` | No | Register user |
| POST | `/api/v1/auth/login` | No | Login user |
| GET | `/api/v1/auth/me` | Yes | Get current user |
| GET | `/api/v1/users/me` | Yes | Get current user profile |
| PATCH | `/api/v1/users/me` | Yes | Update profile |
| POST | `/api/v1/predict/frame` | Yes | Predict one frame |
| GET | `/api/v1/history/predictions` | Yes | List prediction history |
| GET | `/api/v1/history/predictions/{record_id}` | Yes | Get one prediction record |
| POST | `/api/v1/history/phrases` | Yes | Save a phrase |
| GET | `/api/v1/history/phrases` | Yes | List saved phrases |
| GET | `/api/v1/history/phrases/{phrase_id}` | Yes | Get one saved phrase |
| PATCH | `/api/v1/history/phrases/{phrase_id}` | Yes | Update saved phrase |
| DELETE | `/api/v1/history/phrases/{phrase_id}` | Yes | Delete saved phrase |
| POST | `/api/v1/sessions/start` | Yes | Start session |
| POST | `/api/v1/sessions/{session_id}/predict-frame` | Yes | Predict within session |
| POST | `/api/v1/sessions/{session_id}/end` | Yes | End session |
| GET | `/api/v1/sessions` | Yes | List sessions |
| GET | `/api/v1/sessions/{session_id}` | Yes | Get session detail |
| POST | `/api/v1/feedback` | Yes | Submit feedback |
| GET | `/api/v1/feedback/me` | Yes | List my feedback |

---

## Root & Health

### 1) Root
**GET** `/`

**Request Body:** None

**Response Example**
```json
{
  "message": "Arabic Hand Sign Language Backend API",
  "docs": "/docs",
  "openapi": "/openapi.json"
}
```

**Notes**
- Public endpoint
- Useful for quick backend reachability checks

### 2) Service Health
**GET** `/api/v1/health`

**Request Body:** None

**Response Example**
```json
{
  "status": "ok"
}
```

### 3) Database Health
**GET** `/api/v1/health/db`

**Request Body:** None

**Response Example**
```json
{
  "status": "ok",
  "database": "connected"
}
```

**Notes**
- Public endpoint
- Useful for monitoring and app startup diagnostics

---

## Auth Endpoints

### 4) Register User
**POST** `/api/v1/auth/register`

**Request Body**
```json
{
  "email": "user@example.com",
  "password": "StrongPass123",
  "username": "flutter_user",
  "full_name": "Flutter User"
}
```

**Response Example** `201 Created`
```json
{
  "id": "0f9d1f92-8dca-4d00-95f3-4a6e5c2d1abc",
  "email": "user@example.com",
  "username": "flutter_user",
  "full_name": "Flutter User",
  "is_active": true,
  "is_verified": false,
  "created_at": "2026-04-10T08:30:00.000000Z",
  "updated_at": "2026-04-10T08:30:00.000000Z"
}
```

**Notes**
- `password` must be between `8` and `128` chars
- If email already exists → `409`
- If username already exists → `409`

### 5) Login
**POST** `/api/v1/auth/login`

**Request Body**
```json
{
  "email": "user@example.com",
  "password": "StrongPass123"
}
```

**Response Example**
```json
{
  "access_token": "<JWT_TOKEN>",
  "token_type": "bearer"
}
```

**Notes**
- Save `access_token` in secure local storage
- Use it in `Authorization: Bearer <token>`
- Invalid email/password → `401`

### 6) Current Authenticated User
**GET** `/api/v1/auth/me`

**Request Body:** None

**Response Example**
```json
{
  "id": "0f9d1f92-8dca-4d00-95f3-4a6e5c2d1abc",
  "email": "user@example.com",
  "username": "flutter_user",
  "full_name": "Flutter User",
  "is_active": true,
  "is_verified": false,
  "created_at": "2026-04-10T08:30:00.000000Z",
  "updated_at": "2026-04-10T08:30:00.000000Z"
}
```

---

## User Endpoints

### 7) Get My Profile
**GET** `/api/v1/users/me`

**Request Body:** None

**Response Example**
```json
{
  "id": "0f9d1f92-8dca-4d00-95f3-4a6e5c2d1abc",
  "email": "user@example.com",
  "username": "flutter_user",
  "full_name": "Flutter User",
  "is_active": true,
  "is_verified": false,
  "created_at": "2026-04-10T08:30:00.000000Z",
  "updated_at": "2026-04-10T08:30:00.000000Z"
}
```

### 8) Update My Profile
**PATCH** `/api/v1/users/me`

**Request Body**
```json
{
  "username": "new_flutter_user",
  "full_name": "Updated Flutter User"
}
```

**Response Example**
```json
{
  "id": "0f9d1f92-8dca-4d00-95f3-4a6e5c2d1abc",
  "email": "user@example.com",
  "username": "new_flutter_user",
  "full_name": "Updated Flutter User",
  "is_active": true,
  "is_verified": false,
  "created_at": "2026-04-10T08:30:00.000000Z",
  "updated_at": "2026-04-10T09:00:00.000000Z"
}
```

**Notes**
- Send only fields that need updating
- Username conflict → `409`

---

## Prediction Endpoints

### 9) Predict One Frame
**POST** `/api/v1/predict/frame`

**Request Body**
```json
{
  "landmarks": [
    { "x": 0.123, "y": 0.456, "z": -0.012 },
    { "x": 0.130, "y": 0.460, "z": -0.010 },
    { "x": 0.145, "y": 0.470, "z": -0.009 }
  ],
  "top_k": 3,
  "client_timestamp": "2026-04-10T09:10:00Z"
}
```

**Response Example**
```json
{
  "predicted_label": "Alef",
  "arabic_label": "ا",
  "confidence": 0.982341,
  "top_predictions": [
    {
      "label": "Alef",
      "arabic_label": "ا",
      "confidence": 0.982341
    },
    {
      "label": "Beh",
      "arabic_label": "ب",
      "confidence": 0.011245
    },
    {
      "label": "Teh",
      "arabic_label": "ت",
      "confidence": 0.006414
    }
  ],
  "timestamp": "2026-04-10T09:10:01.124000Z"
}
```

**Notes for Flutter**
- `landmarks` must contain **exactly 21 points**
- Each point must include `x`, `y`, `z`
- `top_k` range is `1` to `5`
- This endpoint also stores the prediction into user history
- If landmarks are invalid → `422`

---

## History & Saved Phrases

### 10) List Prediction History
**GET** `/api/v1/history/predictions?skip=0&limit=20&predicted_label=Alef&min_confidence=0.7`

**Request Body:** None

**Query Params**
- `skip` (default `0`)
- `limit` (default `20`, max `100`)
- `predicted_label` (optional)
- `min_confidence` (optional, `0.0` to `1.0`)

**Response Example**
```json
[
  {
    "id": "5e9af0a0-a7a9-4a16-b110-1fe88da92e66",
    "predicted_label": "Alef",
    "arabic_label": "ا",
    "confidence": 0.982341,
    "top_predictions": [
      {
        "label": "Alef",
        "arabic_label": "ا",
        "confidence": 0.982341
      }
    ],
    "client_timestamp": "2026-04-10T09:10:00Z",
    "created_at": "2026-04-10T09:10:01.124000Z"
  }
]
```

### 11) Get One Prediction Record
**GET** `/api/v1/history/predictions/{record_id}`

**Request Body:** None

**Response Example**
```json
{
  "id": "5e9af0a0-a7a9-4a16-b110-1fe88da92e66",
  "predicted_label": "Alef",
  "arabic_label": "ا",
  "confidence": 0.982341,
  "top_predictions": [
    {
      "label": "Alef",
      "arabic_label": "ا",
      "confidence": 0.982341
    }
  ],
  "client_timestamp": "2026-04-10T09:10:00Z",
  "created_at": "2026-04-10T09:10:01.124000Z",
  "raw_landmarks_json": [
    { "x": 0.123, "y": 0.456, "z": -0.012 }
  ]
}
```

### 12) Create Saved Phrase
**POST** `/api/v1/history/phrases`

**Request Body**
```json
{
  "title": "Greeting",
  "content": "السلام عليكم",
  "source_session_id": "8d1df3dd-08b9-4f19-bf3d-9b8d3015e621"
}
```

**Response Example** `201 Created`
```json
{
  "id": "55ecfdf9-2af7-41d5-88fe-e8ea74b0ab11",
  "title": "Greeting",
  "content": "السلام عليكم",
  "source_session_id": "8d1df3dd-08b9-4f19-bf3d-9b8d3015e621",
  "created_at": "2026-04-10T09:20:00.000000Z",
  "updated_at": "2026-04-10T09:20:00.000000Z"
}
```

### 13) List Saved Phrases
**GET** `/api/v1/history/phrases?skip=0&limit=20`

**Request Body:** None

**Response Example**
```json
[
  {
    "id": "55ecfdf9-2af7-41d5-88fe-e8ea74b0ab11",
    "title": "Greeting",
    "content": "السلام عليكم",
    "source_session_id": "8d1df3dd-08b9-4f19-bf3d-9b8d3015e621",
    "created_at": "2026-04-10T09:20:00.000000Z",
    "updated_at": "2026-04-10T09:20:00.000000Z"
  }
]
```

### 14) Get One Saved Phrase
**GET** `/api/v1/history/phrases/{phrase_id}`

**Request Body:** None

**Response Example**
```json
{
  "id": "55ecfdf9-2af7-41d5-88fe-e8ea74b0ab11",
  "title": "Greeting",
  "content": "السلام عليكم",
  "source_session_id": "8d1df3dd-08b9-4f19-bf3d-9b8d3015e621",
  "created_at": "2026-04-10T09:20:00.000000Z",
  "updated_at": "2026-04-10T09:20:00.000000Z"
}
```

### 15) Update Saved Phrase
**PATCH** `/api/v1/history/phrases/{phrase_id}`

**Request Body**
```json
{
  "title": "Greeting Updated",
  "content": "السلام عليكم ورحمة الله"
}
```

**Response Example**
```json
{
  "id": "55ecfdf9-2af7-41d5-88fe-e8ea74b0ab11",
  "title": "Greeting Updated",
  "content": "السلام عليكم ورحمة الله",
  "source_session_id": "8d1df3dd-08b9-4f19-bf3d-9b8d3015e621",
  "created_at": "2026-04-10T09:20:00.000000Z",
  "updated_at": "2026-04-10T09:25:00.000000Z"
}
```

### 16) Delete Saved Phrase
**DELETE** `/api/v1/history/phrases/{phrase_id}`

**Request Body:** None

**Response**
- `204 No Content`

**Notes**
- No response body on success

---

## Session Endpoints

### 17) Start Session
**POST** `/api/v1/sessions/start`

**Request Body**
```json
{
  "notes": "Practice session for alphabet recognition"
}
```

**Response Example** `201 Created`
```json
{
  "id": "8d1df3dd-08b9-4f19-bf3d-9b8d3015e621",
  "status": "active",
  "notes": "Practice session for alphabet recognition",
  "started_at": "2026-04-10T09:30:00.000000Z",
  "ended_at": null,
  "prediction_count": 0
}
```

### 18) Predict Inside Session
**POST** `/api/v1/sessions/{session_id}/predict-frame`

**Request Body**
```json
{
  "landmarks": [
    { "x": 0.123, "y": 0.456, "z": -0.012 },
    { "x": 0.130, "y": 0.460, "z": -0.010 },
    { "x": 0.145, "y": 0.470, "z": -0.009 }
  ],
  "top_k": 3,
  "client_timestamp": "2026-04-10T09:31:00Z"
}
```

**Response Example**
```json
{
  "predicted_label": "Beh",
  "arabic_label": "ب",
  "confidence": 0.9542,
  "top_predictions": [
    {
      "label": "Beh",
      "arabic_label": "ب",
      "confidence": 0.9542
    },
    {
      "label": "Teh",
      "arabic_label": "ت",
      "confidence": 0.0301
    }
  ],
  "timestamp": "2026-04-10T09:31:01.000000Z"
}
```

**Notes**
- Session must exist and belong to the logged-in user
- Session must still be `active`
- If already ended → `409`

### 19) End Session
**POST** `/api/v1/sessions/{session_id}/end`

**Request Body:** None

**Response Example**
```json
{
  "id": "8d1df3dd-08b9-4f19-bf3d-9b8d3015e621",
  "status": "completed",
  "notes": "Practice session for alphabet recognition",
  "started_at": "2026-04-10T09:30:00.000000Z",
  "ended_at": "2026-04-10T09:40:00.000000Z",
  "prediction_count": 12
}
```

### 20) List Sessions
**GET** `/api/v1/sessions?skip=0&limit=20`

**Request Body:** None

**Response Example**
```json
[
  {
    "id": "8d1df3dd-08b9-4f19-bf3d-9b8d3015e621",
    "status": "completed",
    "notes": "Practice session for alphabet recognition",
    "started_at": "2026-04-10T09:30:00.000000Z",
    "ended_at": "2026-04-10T09:40:00.000000Z",
    "prediction_count": 12
  }
]
```

### 21) Get Session Detail
**GET** `/api/v1/sessions/{session_id}`

**Request Body:** None

**Response Example**
```json
{
  "id": "8d1df3dd-08b9-4f19-bf3d-9b8d3015e621",
  "status": "completed",
  "notes": "Practice session for alphabet recognition",
  "started_at": "2026-04-10T09:30:00.000000Z",
  "ended_at": "2026-04-10T09:40:00.000000Z",
  "prediction_count": 12,
  "recent_predictions": [
    {
      "id": "5e9af0a0-a7a9-4a16-b110-1fe88da92e66",
      "predicted_label": "Alef",
      "arabic_label": "ا",
      "confidence": 0.982341,
      "top_predictions": [
        {
          "label": "Alef",
          "arabic_label": "ا",
          "confidence": 0.982341
        }
      ],
      "client_timestamp": "2026-04-10T09:10:00Z",
      "created_at": "2026-04-10T09:10:01.124000Z"
    }
  ]
}
```

---

## Feedback Endpoints

### 22) Create Feedback
**POST** `/api/v1/feedback`

**Request Body**
```json
{
  "prediction_record_id": "5e9af0a0-a7a9-4a16-b110-1fe88da92e66",
  "session_id": null,
  "is_correct": false,
  "expected_label": "Teh",
  "notes": "The hand sign was closer to Teh than Beh"
}
```

**Response Example** `201 Created`
```json
{
  "id": "7d38db1d-cd27-456a-8385-bd2a363adf11",
  "prediction_record_id": "5e9af0a0-a7a9-4a16-b110-1fe88da92e66",
  "session_id": null,
  "is_correct": false,
  "expected_label": "Teh",
  "notes": "The hand sign was closer to Teh than Beh",
  "created_at": "2026-04-10T09:45:00.000000Z"
}
```

**Notes**
- At least one of these must be provided:
  - `prediction_record_id`
  - `session_id`
- Both are allowed if relevant
- If referenced record/session does not belong to user → `404`

### 23) List My Feedback
**GET** `/api/v1/feedback/me?skip=0&limit=20`

**Request Body:** None

**Response Example**
```json
[
  {
    "id": "7d38db1d-cd27-456a-8385-bd2a363adf11",
    "prediction_record_id": "5e9af0a0-a7a9-4a16-b110-1fe88da92e66",
    "session_id": null,
    "is_correct": false,
    "expected_label": "Teh",
    "notes": "The hand sign was closer to Teh than Beh",
    "created_at": "2026-04-10T09:45:00.000000Z"
  }
]
```

---

## Flutter Integration Notes

### Recommended client setup
- Use `Dio` or `http` package
- Store JWT using `flutter_secure_storage`
- Add an interceptor to inject the bearer token automatically
- On `401`, clear token and redirect user to login

### Prediction payload notes
- Send **exactly 21 landmarks** per request
- Keep landmark order identical to MediaPipe hand landmark order
- `client_timestamp` should be generated on device in ISO format

### UI/UX notes
- Show `arabic_label` directly in UI for Arabic output
- Use `top_predictions` to display confidence breakdown or alternatives
- For saved phrases and session history, cache the UUIDs locally for future detail/update/delete calls

### Useful links
- Swagger UI: `https://ahs-production-7427.up.railway.app/docs`
- OpenAPI JSON: `https://ahs-production-7427.up.railway.app/openapi.json`
