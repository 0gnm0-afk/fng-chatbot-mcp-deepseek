"""Public error types for CNN data retrieval and normalization."""

from __future__ import annotations


class CnnClientError(RuntimeError):
    """Base class for expected CNN client failures."""


class CnnTimeoutError(CnnClientError):
    """The CNN request exceeded its configured timeout."""


class CnnNetworkError(CnnClientError):
    """The CNN request failed before a valid HTTP response arrived."""


class CnnBlockedError(CnnClientError):
    """CNN rejected or rate-limited the request."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"CNN request was blocked or rate-limited (HTTP {status_code})")


class CnnHttpError(CnnClientError):
    """CNN returned a non-success HTTP status."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"CNN request failed with HTTP {status_code}")


class CnnPayloadError(CnnClientError):
    """CNN returned a response body that is not a usable JSON object."""


class CnnSchemaError(CnnClientError):
    """CNN JSON did not satisfy the project's normalization contract."""
