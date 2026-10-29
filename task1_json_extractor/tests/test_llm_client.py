"""Provider client tests with mocked SDK clients. No network is used."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from config import Settings
from exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from llm_client import GeminiClient, OpenAIClient, build_llm_client
from schemas import ResumeExtraction


def _settings(provider: str = "gemini") -> Settings:
    return Settings(
        provider=provider,  # type: ignore[arg-type]
        model="test-model",
        timeout_seconds=10.0,
        max_output_retries=1,
        api_key="sk-secret-value-that-must-not-leak",
    )


def test_build_llm_client_selects_configured_provider() -> None:
    gemini = build_llm_client(_settings("gemini"))
    openai = build_llm_client(_settings("openai"))

    assert isinstance(gemini, GeminiClient)
    assert isinstance(openai, OpenAIClient)


def test_settings_summary_never_contains_api_key() -> None:
    settings = _settings()

    assert "sk-secret" not in settings.public_summary()
    assert "sk-secret" not in repr(settings)


def test_gemini_client_requests_native_json_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = GeminiClient(_settings("gemini"))
    captured: dict[str, Any] = {}

    def fake_generate_content(
        *, model: str, contents: str, config: Any
    ) -> SimpleNamespace:
        captured["model"] = model
        captured["contents"] = contents
        captured["config"] = config
        return SimpleNamespace(
            text='{"full_name": "Jane Doe"}',
            candidates=[],
            prompt_feedback=None,
        )

    monkeypatch.setattr(
        client._client.models, "generate_content", fake_generate_content
    )

    raw = client.generate_json(
        system_prompt="System rules",
        user_prompt="Document text",
        response_model=ResumeExtraction,
    )

    assert raw == '{"full_name": "Jane Doe"}'
    assert captured["model"] == "test-model"
    assert captured["contents"] == "Document text"
    config = captured["config"]
    assert config.system_instruction == "System rules"
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema["type"] == "object"
    assert config.response_json_schema["additionalProperties"] is False


def test_openai_client_requests_strict_json_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OpenAIClient(_settings("openai"))
    captured: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content='{"full_name": "Jane Doe"}', refusal=None
                    ),
                )
            ]
        )

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

    raw = client.generate_json(
        system_prompt="System rules",
        user_prompt="Document text",
        response_model=ResumeExtraction,
    )

    assert raw == '{"full_name": "Jane Doe"}'
    assert captured["model"] == "test-model"
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert (
        captured["response_format"]["json_schema"]["schema"][
            "additionalProperties"
        ]
        is False
    )
    assert captured["messages"][0]["role"] == "system"


def test_openai_refusal_is_reported_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exceptions import LLMProviderResponseError

    client = OpenAIClient(_settings("openai"))

    def fake_create(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=None, refusal="cannot comply"),
                )
            ]
        )

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

    with pytest.raises(LLMProviderResponseError, match="refused"):
        client.generate_json(
            system_prompt="System",
            user_prompt="Document",
            response_model=ResumeExtraction,
        )


def test_openai_authentication_error_is_translated_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx
    import openai

    client = OpenAIClient(_settings("openai"))
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(401, request=request)

    def fake_create(**kwargs: Any) -> None:
        raise openai.AuthenticationError(
            "Incorrect API key provided: sk-secret-value-that-must-not-leak",
            response=response,
            body=None,
        )

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

    with pytest.raises(LLMAuthenticationError) as excinfo:
        client.generate_json(
            system_prompt="System",
            user_prompt="Document",
            response_model=ResumeExtraction,
        )

    assert "sk-secret" not in str(excinfo.value)


def test_gemini_rate_limit_error_is_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.genai import errors

    client = GeminiClient(_settings("gemini"))

    def fake_generate_content(**kwargs: Any) -> None:
        raise errors.APIError(429, {"error": {"message": "quota exceeded"}})

    monkeypatch.setattr(
        client._client.models, "generate_content", fake_generate_content
    )

    with pytest.raises(LLMRateLimitError):
        client.generate_json(
            system_prompt="System",
            user_prompt="Document",
            response_model=ResumeExtraction,
        )


def test_gemini_timeout_error_is_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    client = GeminiClient(_settings("gemini"))

    def fake_generate_content(**kwargs: Any) -> None:
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(
        client._client.models, "generate_content", fake_generate_content
    )

    with pytest.raises(LLMTimeoutError):
        client.generate_json(
            system_prompt="System",
            user_prompt="Document",
            response_model=ResumeExtraction,
        )


def test_gemini_retired_model_error_includes_google_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exceptions import LLMRequestError
    from google.genai import errors

    client = GeminiClient(_settings("gemini"))

    def fake_generate_content(**kwargs: Any) -> None:
        raise errors.APIError(
            404,
            {
                "error": {
                    "code": 404,
                    "message": (
                        "This model models/gemini-2.5-flash is no longer "
                        "available to new users. Please update your code to "
                        "use models/gemini-3.8-flash."
                    ),
                    "status": "NOT_FOUND",
                }
            },
        )

    monkeypatch.setattr(
        client._client.models, "generate_content", fake_generate_content
    )

    with pytest.raises(LLMRequestError) as excinfo:
        client.generate_json(
            system_prompt="System",
            user_prompt="Document",
            response_model=ResumeExtraction,
        )

    message = str(excinfo.value)
    assert "no longer available" in message
    assert "gemini-3.8-flash" in message
    assert "sk-secret" not in message
