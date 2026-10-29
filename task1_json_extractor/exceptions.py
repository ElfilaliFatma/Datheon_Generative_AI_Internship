"""Application-specific exceptions.

The messages on these exceptions are safe to show to a user. Provider error
bodies and document content are deliberately not copied into exception text.
"""

from __future__ import annotations


class ApplicationError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(ApplicationError):
    """The application is not configured correctly."""


class InputError(ApplicationError):
    """A document or file path is missing, unreadable, or invalid."""


class OutputWriteError(ApplicationError):
    """A JSON result could not be written safely."""


class ExtractionError(ApplicationError):
    """Base class for extraction failures."""

    def __init__(self, message: str, *, attempts: int = 1) -> None:
        super().__init__(message)
        self.attempts = attempts


class OutputFormatError(ExtractionError):
    """The model output was empty, malformed, or did not match the schema."""


class EmptyResponseError(OutputFormatError):
    """The model returned no usable text."""


class InvalidJSONError(OutputFormatError):
    """The model response did not contain a JSON object."""


class SchemaValidationError(OutputFormatError):
    """Parsed JSON did not satisfy the selected Pydantic schema."""

    def __init__(
        self,
        message: str,
        *,
        fields: tuple[str, ...] = (),
        attempts: int = 1,
    ) -> None:
        super().__init__(message, attempts=attempts)
        self.fields = fields


class LLMError(ExtractionError):
    """Base class for failures reported by an LLM provider."""


class LLMAuthenticationError(LLMError):
    """The provider rejected the credential or permission."""


class LLMRateLimitError(LLMError):
    """The provider rate limit or quota was reached."""


class LLMTimeoutError(LLMError):
    """The provider request timed out."""


class LLMConnectionError(LLMError):
    """The provider could not be reached."""


class LLMRequestError(LLMError):
    """The provider rejected the request (for example, an unsupported model)."""


class LLMProviderResponseError(LLMError):
    """The provider returned an unusable or refused response."""
