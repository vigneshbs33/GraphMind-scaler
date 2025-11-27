"""Custom exceptions for GraphMind."""

from typing import Optional


class GraphMindError(Exception):
    """Base exception for GraphMind."""

    def __init__(self, message: str, status_code: int = 500, details: Optional[dict] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class StorageError(GraphMindError):
    """Storage-related errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, status_code=500, details=details)


class ValidationError(GraphMindError):
    """Input validation errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, status_code=400, details=details)


class NotFoundError(GraphMindError):
    """Resource not found errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, status_code=404, details=details)


class LLMError(GraphMindError):
    """LLM-related errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, status_code=503, details=details)

