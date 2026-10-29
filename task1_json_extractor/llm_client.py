"""Provider-specific LLM clients using native structured-output features.

Only the provider selected in ``Settings`` is used. Failures never fall back
to another provider, and safe error messages never include credentials or
document content.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx
from pydantic import BaseModel

from config import Settings
from exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMProviderResponseError,
    LLMRateLimitError,
    LLMRequestError,
    LLMTimeoutError,
)
from schemas import build_provider_json_schema


@runtime_checkable
class JSONModelClient(Protocol):
    """Interface used by the extractor and implemented by test doubles."""

    provider: str
    model: str

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> str:
        """Return raw JSON text conforming (ideally) to ``response_model``."""


class GeminiClient:
    """Google Gemini client using the google-genai SDK."""

    provider = "gemini"

    def __init__(self, settings: Settings) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - installed via requirements
            raise LLMRequestError(
                "The google-genai package is not installed."
            ) from exc

        self.model = settings.model
        self._timeout_seconds = settings.timeout_seconds
        self._types = types
        try:
            self._client = genai.Client(
                api_key=settings.api_key,
                http_options=types.HttpOptions(
                    timeout=int(settings.timeout_seconds * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        except Exception:
            raise LLMRequestError(
                "Could not initialize the Google Gemini client. "
                "Check the SDK installation and configuration."
            ) from None

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> str:
        schema = build_provider_json_schema(response_model)
        try:
            config = self._types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
                response_mime_type="application/json",
                response_json_schema=schema,
            )
        except (TypeError, ValueError):
            raise LLMRequestError(
                "The response schema could not be converted for the Gemini API."
            ) from None

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config,
            )
        except Exception as exc:
            _raise_gemini_error(exc, timeout_seconds=self._timeout_seconds)
            raise

        _check_gemini_finish_state(response)
        try:
            text = response.text
        except (ValueError, AttributeError):
            raise LLMProviderResponseError(
                "Gemini returned no usable text response."
            ) from None
        return text or ""


class OpenAIClient:
    """OpenAI client using strict JSON Schema structured outputs."""

    provider = "openai"

    def __init__(self, settings: Settings) -> None:
        try:
            import openai
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - installed via requirements
            raise LLMRequestError("The openai package is not installed.") from exc

        self._openai = openai
        self.model = settings.model
        try:
            self._client = OpenAI(
                api_key=settings.api_key,
                timeout=settings.timeout_seconds,
                max_retries=0,
            )
        except Exception:
            raise LLMRequestError(
                "Could not initialize the OpenAI client. "
                "Check the SDK installation and configuration."
            ) from None

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> str:
        schema = build_provider_json_schema(response_model)
        schema_name = _openai_schema_name(response_model)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
                temperature=0.0,
            )
        except Exception as exc:
            _raise_openai_error(exc, self._openai)
            raise

        if not response.choices:
            raise LLMProviderResponseError(
                "OpenAI returned a response with no choices."
            )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMProviderResponseError(
                "OpenAI stopped because the output token limit was reached."
            )
        if choice.finish_reason == "content_filter":
            raise LLMProviderResponseError(
                "OpenAI blocked the response due to a content filter."
            )
        message = choice.message
        if getattr(message, "refusal", None):
            raise LLMProviderResponseError(
                "OpenAI refused to produce a structured response."
            )
        return message.content or ""


def build_llm_client(settings: Settings) -> JSONModelClient:
    """Instantiate exactly the configured provider. Never fall back."""
    if settings.provider == "gemini":
        return GeminiClient(settings)
    if settings.provider == "openai":
        return OpenAIClient(settings)
    raise LLMRequestError(
        "Unsupported LLM_PROVIDER. Choose 'gemini' or 'openai'."
    )


def _openai_schema_name(response_model: type[BaseModel]) -> str:
    name = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in response_model.__name__
    ).strip("_")
    return (name or "extraction")[:64]


def _check_gemini_finish_state(response: object) -> None:
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(feedback, "block_reason", None) if feedback else None
    if block_reason:
        raise LLMProviderResponseError(
            "Gemini blocked the prompt before producing a response."
        )

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        finish_reason = getattr(candidate, "finish_reason", None)
        finish_name = getattr(finish_reason, "name", str(finish_reason or "")).upper()
        if finish_name in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"}:
            raise LLMProviderResponseError(
                "Gemini stopped generation for safety or policy reasons."
            )
        if finish_name == "MAX_TOKENS":
            raise LLMProviderResponseError(
                "Gemini stopped because the output token limit was reached."
            )
        if finish_name == "MALFORMED_FUNCTION_CALL":
            raise LLMProviderResponseError(
                "Gemini produced an unusable function call."
            )


def _raise_gemini_error(exc: Exception, *, timeout_seconds: float) -> None:
    """Translate Gemini/HTTP errors into safe application exceptions."""
    try:
        from google.genai import errors
    except ImportError:  # pragma: no cover
        errors = None  # type: ignore[assignment]

    if errors is not None and isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        if code in {401, 403}:
            raise LLMAuthenticationError(
                "Gemini authentication or permission failed. Check "
                "GEMINI_API_KEY and access to the configured model."
            ) from None
        if code == 429:
            raise LLMRateLimitError(
                "Gemini rate limit or quota exceeded. Wait before retrying."
            ) from None
        if code in {408, 504}:
            raise LLMTimeoutError(
                "The Gemini request timed out. Try again later."
            ) from None
        if code in {400, 404, 422}:
            raise LLMRequestError(
                f"Gemini rejected the request (HTTP {code}). Check the "
                "selected model and structured-output support."
            ) from None
        if code and code >= 500:
            raise LLMProviderResponseError(
                f"Gemini is temporarily unavailable (HTTP {code})."
            ) from None
        raise LLMProviderResponseError("Gemini returned an API error.") from None

    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        raise LLMTimeoutError(
            f"The Gemini request timed out after {timeout_seconds:g} seconds."
        ) from None
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError, ConnectionError)):
        raise LLMConnectionError(
            "Could not connect to the Gemini API. Check network access."
        ) from None
    raise LLMProviderResponseError(
        "The Gemini request failed due to an unexpected client error."
    ) from None


def _raise_openai_error(exc: Exception, openai_module: object) -> None:
    """Translate OpenAI SDK errors into safe application exceptions."""
    if isinstance(exc, getattr(openai_module, "AuthenticationError")):
        raise LLMAuthenticationError(
            "OpenAI authentication failed. Check OPENAI_API_KEY."
        ) from None
    if isinstance(exc, getattr(openai_module, "PermissionDeniedError")):
        raise LLMAuthenticationError(
            "OpenAI permission denied for the configured model or project."
        ) from None
    if isinstance(exc, getattr(openai_module, "RateLimitError")):
        raise LLMRateLimitError(
            "OpenAI rate limit or quota exceeded. Wait before retrying."
        ) from None
    if isinstance(exc, getattr(openai_module, "APITimeoutError")):
        raise LLMTimeoutError("The OpenAI request timed out. Try again later.") from None
    if isinstance(exc, getattr(openai_module, "APIConnectionError")):
        raise LLMConnectionError(
            "Could not connect to the OpenAI API. Check network access."
        ) from None
    if isinstance(exc, getattr(openai_module, "APIStatusError")):
        status_code = getattr(exc, "status_code", None)
        raise LLMRequestError(
            f"OpenAI rejected the request (HTTP {status_code}). Check the "
            "selected model and structured-output support."
        ) from None
    if isinstance(exc, getattr(openai_module, "OpenAIError")):
        raise LLMProviderResponseError("OpenAI returned an API error.") from None
    raise LLMProviderResponseError(
        "The OpenAI request failed due to an unexpected client error."
    ) from None
