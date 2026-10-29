"""Configuration validation tests. No real credentials are used."""

from __future__ import annotations

from pathlib import Path

import pytest

import config as config_module
from config import Settings
from exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def clean_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ignore any developer .env so tests never depend on local values."""
    monkeypatch.setattr(
        config_module, "ENV_FILE", tmp_path / "no-such-file.env"
    )
    for name in (
        "LLM_PROVIDER",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_MODEL",
        "OPENAI_MODEL",
        "LLM_TIMEOUT_SECONDS",
        "LLM_MAX_OUTPUT_RETRIES",
    ):
        monkeypatch.delenv(name, raising=False)


def test_missing_provider_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="LLM_PROVIDER"):
        Settings.from_env()


def test_invalid_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    with pytest.raises(ConfigurationError, match="gemini.*openai"):
        Settings.from_env()


def test_gemini_requires_only_gemini_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

    settings = Settings.from_env()

    assert settings.provider == "gemini"
    assert settings.model == "gemini-3.5-flash"
    assert settings.api_key == "fake-gemini-key"


def test_openai_requires_only_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model-name")

    settings = Settings.from_env()

    assert settings.provider == "openai"
    assert settings.model == "test-model-name"
    assert settings.api_key == "fake-openai-key"


def test_missing_selected_provider_key_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("GEMINI_API_KEY", "present-but-irrelevant")

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        Settings.from_env()


def test_blank_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "   ")

    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        Settings.from_env()


@pytest.mark.parametrize(
    "placeholder",
    ["your_gemini_api_key_here", "YOUR_OPENAI_API_KEY_HERE", "changeme"],
)
def test_placeholder_api_key_is_rejected(
    monkeypatch: pytest.MonkeyPatch, placeholder: str
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", placeholder)

    with pytest.raises(ConfigurationError, match="placeholder"):
        Settings.from_env()


@pytest.mark.parametrize("model", ["bad model!", "-leading", "a" * 200])
def test_invalid_model_names_are_rejected(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MODEL", model)

    with pytest.raises(ConfigurationError, match="GEMINI_MODEL"):
        Settings.from_env()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LLM_TIMEOUT_SECONDS", "abc"),
        ("LLM_TIMEOUT_SECONDS", "1"),
        ("LLM_MAX_OUTPUT_RETRIES", "9"),
        ("LLM_MAX_OUTPUT_RETRIES", "1.5"),
    ],
)
def test_invalid_limits_are_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError):
        Settings.from_env()


def test_api_key_is_hidden_from_repr_and_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key")

    settings = Settings.from_env()

    assert "super-secret-key" not in repr(settings)
    assert "super-secret-key" not in settings.public_summary()
