from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings
from app.core.constants import API_DESCRIPTION, API_TAGS
from app.db.init_db import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title=settings.APP_NAME,
        description=API_DESCRIPTION,
        version="0.2.0",
        debug=settings.DEBUG,
        openapi_tags=API_TAGS,
        lifespan=lifespan,
    )

    application.include_router(api_router)

    @application.get("/", tags=["root"])
    def read_root() -> dict[str, str]:
        return {
            "message": settings.APP_NAME,
            "docs": "/docs",
            "openapi": "/openapi.json",
        }

    return application


app = create_application()
