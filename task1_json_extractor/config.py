"""Environment-based application configuration.

Only the API key for the explicitly selected provider is read. Configuration
objects never expose the key through their repr or summary helpers.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from exceptions import ConfigurationError

Provider = Literal["gemini", "openai"]

SUPPORTED_PROVIDERS: tuple[Provider, ...] = ("gemini", "openai")
DEFAULT_MODELS: dict[Provider, str] = {
    "gemini": "gemini-3.5-flash",
    "openai": "gpt-4o-mini",
}
API_KEY_ENV_VARS: dict[Provider, str] = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}
MODEL_ENV_VARS: dict[Provider, str] = {
    "gemini": "GEMINI_MODEL",
    "openai": "OPENAI_MODEL",
}

ENV_FILE = Path(__file__).resolve().parent / ".env"
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_PLACEHOLDER_SECRETS = {
    "your_gemini_api_key_here",
    "your_openai_api_key_here",
    "your_api_key_here",
    "replace_me",
    "changeme",
}


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime settings for a single provider."""

    provider: Provider
    model: str
    timeout_seconds: float
    max_output_retries: int
    api_key: str = field(repr=False, compare=False)

    @property
    def provider_label(self) -> str:
        return "Google Gemini" if self.provider == "gemini" else "OpenAI"

    @property
    def api_key_env_var(self) -> str:
        return API_KEY_ENV_VARS[self.provider]

    def public_summary(self) -> str:
        """Return configuration details that are safe to display."""
        return (
            f"provider={self.provider}, model={self.model}, "
            f"timeout={self.timeout_seconds:g}s, "
            f"max_output_retries={self.max_output_retries}"
        )

    @classmethod
    def from_env(cls) -> "Settings":
        """Load and validate settings from the process environment and ``.env``.

        A ``.env`` file beside this module is read when python-dotenv is
        available. Existing environment variables take precedence.
        """
        _load_dotenv_file()

        raw_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        if not raw_provider:
            raise ConfigurationError(
                "LLM_PROVIDER is not set. Set it to 'gemini' or 'openai' "
                "(for example, in .env)."
            )
        if raw_provider not in SUPPORTED_PROVIDERS:
            raise ConfigurationError(
                "LLM_PROVIDER must be either 'gemini' or 'openai'."
            )
        provider: Provider = raw_provider  # type: ignore[assignment]

        key_env_var = API_KEY_ENV_VARS[provider]
        api_key = os.getenv(key_env_var, "").strip()
        if not api_key:
            raise ConfigurationError(
                f"{key_env_var} is not set. Configure the API key for the "
                f"selected provider ({provider}) in .env or the environment."
            )
        if api_key.casefold() in _PLACEHOLDER_SECRETS:
            raise ConfigurationError(
                f"{key_env_var} still contains the placeholder value from "
                ".env.example. Replace it with a real API key or unset it."
            )

        model_env_var = MODEL_ENV_VARS[provider]
        model = (os.getenv(model_env_var) or DEFAULT_MODELS[provider]).strip()
        if not _MODEL_NAME_RE.fullmatch(model):
            raise ConfigurationError(
                f"{model_env_var} must be a valid model identifier "
                "(letters, digits, '.', '_', ':', '/', or '-')."
            )

        timeout_seconds = _read_number(
            "LLM_TIMEOUT_SECONDS", default=60.0, minimum=5.0, maximum=300.0
        )
        max_output_retries = _read_int(
            "LLM_MAX_OUTPUT_RETRIES", default=1, minimum=0, maximum=3
        )

        return cls(
            provider=provider,
            model=model,
            timeout_seconds=timeout_seconds,
            max_output_retries=max_output_retries,
            api_key=api_key,
        )


def _load_dotenv_file() -> None:
    if not ENV_FILE.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - dependency is installed
        raise ConfigurationError(
            "python-dotenv is not installed. Install requirements.txt."
        ) from exc
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def _read_number(
    name: str, *, default: float, minimum: float, maximum: float
) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number.") from exc
    if not (minimum <= value <= maximum):
        raise ConfigurationError(
            f"{name} must be between {minimum:g} and {maximum:g}."
        )
    return value


def _read_int(name: str, *, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a whole number.") from exc
    if not (minimum <= value <= maximum):
        raise ConfigurationError(
            f"{name} must be between {minimum} and {maximum}."
        )
    return value
