class AppException(Exception):
    """Base exception for backend application errors."""


class DatabaseConnectionError(AppException):
    """Raised when the application cannot reach the configured database."""
