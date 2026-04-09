from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings
from app.core.constants import API_DESCRIPTION, API_TAGS


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title=settings.APP_NAME,
        description=API_DESCRIPTION,
        version="0.1.0",
        debug=settings.DEBUG,
        openapi_tags=API_TAGS,
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
