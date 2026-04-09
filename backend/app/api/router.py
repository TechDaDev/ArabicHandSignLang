from fastapi import APIRouter

from app.api.v1 import auth, feedback, history, predict, sessions, users


api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(predict.router)
api_router.include_router(sessions.router)
api_router.include_router(history.router)
api_router.include_router(feedback.router)
