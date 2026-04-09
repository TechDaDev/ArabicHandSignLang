class AppError(Exception):
    """Base application error for clean API responses."""


class InvalidLandmarkPayload(AppError):
    """Raised when the landmark request body is malformed."""


class ModelArtifactsUnavailable(AppError):
    """Raised when the ML artifacts cannot be loaded."""
