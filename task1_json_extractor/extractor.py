"""Structured extraction pipeline with parsing, validation, and safe retries."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from config import Settings
from exceptions import (
    EmptyResponseError,
    InvalidJSONError,
    OutputFormatError,
    SchemaValidationError,
)
from json_io import serialize_model, validate_document_text
from llm_client import JSONModelClient, build_llm_client
from prompts import build_extraction_prompt
from schemas import ExtractionMode, schema_for_mode

_CODE_FENCE_RE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n?(.*?)\r?\n?```\Z",
    re.IGNORECASE | re.DOTALL,
)

_RETRY_FEEDBACK = (
    "\n\nCORRECTION REQUIRED: The previous response could not be parsed or "
    "validated. Return one complete JSON object only, with no markdown or "
    "commentary. Follow the schema exactly. Use null for missing scalar "
    "values and [] for missing collections. Use JSON numbers for numeric "
    "fields and YYYY-MM-DD for full dates."
)


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """A validated extraction and its non-sensitive execution metadata."""

    mode: ExtractionMode
    data: BaseModel
    provider: str
    model: str
    attempts: int

    def to_json(self) -> str:
        """Serialize the validated data as UTF-8-ready JSON text."""
        return serialize_model(self.data)

    def summary(self) -> str:
        """Return a short summary that avoids exposing extracted PII."""
        if self.mode is ExtractionMode.RESUME:
            resume = self.data
            return (
                "resume: "
                f"education={len(getattr(resume, 'education', []))}, "
                f"work_experience={len(getattr(resume, 'work_experience', []))}, "
                f"projects={len(getattr(resume, 'projects', []))}, "
                f"technical_skills={len(getattr(resume, 'technical_skills', []))}, "
                f"certifications={len(getattr(resume, 'certifications', []))}"
            )
        invoice = self.data
        total = getattr(invoice, "total_amount", None)
        currency = getattr(invoice, "currency", None)
        amount = "unknown" if total is None else f"{total:f}"
        return (
            "invoice: "
            f"line_items={len(getattr(invoice, 'line_items', []))}, "
            f"total={amount}, currency={currency or 'unknown'}"
        )


class Extractor:
    """Run the prompt -> LLM -> parse -> validate -> serialize pipeline."""

    def __init__(
        self,
        settings: Settings,
        client: JSONModelClient | None = None,
    ) -> None:
        self._settings = settings
        self._client = client if client is not None else build_llm_client(settings)

    @property
    def client(self) -> JSONModelClient:
        return self._client

    def extract(
        self, mode: ExtractionMode | str, document: str
    ) -> ExtractionResult:
        """Extract and validate one document.

        Only output-format failures are retried, up to
        ``Settings.max_output_retries`` extra attempts. Provider errors fail
        immediately with a safe, typed exception.
        """
        try:
            normalized_mode = ExtractionMode(mode)
        except ValueError:
            raise ValueError("mode must be 'resume' or 'invoice'") from None

        text = validate_document_text(document)
        prompts = build_extraction_prompt(normalized_mode, text)
        response_model = schema_for_mode(normalized_mode)
        max_attempts = 1 + self._settings.max_output_retries
        last_format_error: OutputFormatError | None = None

        for attempt in range(1, max_attempts + 1):
            user_prompt = prompts.user
            if attempt > 1:
                user_prompt += _RETRY_FEEDBACK

            raw = self._client.generate_json(
                system_prompt=prompts.system,
                user_prompt=user_prompt,
                response_model=response_model,
            )
            try:
                payload = parse_json_object(raw)
                data = _validate_payload(response_model, payload)
                _verify_json_round_trip(data)
            except OutputFormatError as exc:
                exc.attempts = attempt
                last_format_error = exc
                if attempt < max_attempts:
                    continue
                raise

            return ExtractionResult(
                mode=normalized_mode,
                data=data,
                provider=self._client.provider,
                model=self._client.model,
                attempts=attempt,
            )

        # Defensive: the loop always returns or raises.
        raise last_format_error or OutputFormatError(
            "Extraction failed without a result."
        )


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse one JSON object, tolerating a single surrounding code fence."""
    if not isinstance(raw, str):
        raise InvalidJSONError("The model response was not text.")
    text = raw.lstrip("\ufeff").strip()
    if not text:
        raise EmptyResponseError("The model returned an empty response.")

    fence_match = _CODE_FENCE_RE.fullmatch(text)
    if fence_match:
        text = fence_match.group(1).strip()
        if not text:
            raise EmptyResponseError("The model returned an empty JSON block.")

    if not text.startswith("{"):
        raise InvalidJSONError(
            "The model response did not contain a JSON object."
        )

    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-finite JSON number: {constant}")

    try:
        payload = json.loads(text, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise InvalidJSONError(
            f"Malformed JSON at line {exc.lineno}, column {exc.colno}."
        ) from None
    except ValueError:
        raise InvalidJSONError(
            "Malformed JSON: non-finite numbers are not allowed."
        ) from None

    if not isinstance(payload, dict):
        raise InvalidJSONError("The model response JSON was not an object.")
    return payload


def _validate_payload(
    response_model: type[BaseModel], payload: dict[str, Any]
) -> BaseModel:
    try:
        return response_model.model_validate(payload)
    except ValidationError as exc:
        locations: list[str] = []
        for error in exc.errors()[:8]:
            location = ".".join(str(part) for part in error.get("loc", ()))
            error_type = str(error.get("type", "invalid"))
            label = location or "<root>"
            locations.append(f"{label} ({error_type})")
        fields = tuple(locations)
        detail = ", ".join(fields) if fields else "unknown fields"
        raise SchemaValidationError(
            f"Extracted data failed schema validation: {detail}.",
            fields=fields,
        ) from None


def _verify_json_round_trip(data: BaseModel) -> None:
    try:
        serialize_model(data)
    except (TypeError, ValueError):
        raise SchemaValidationError(
            "Validated data could not be serialized as strict JSON."
        ) from None
