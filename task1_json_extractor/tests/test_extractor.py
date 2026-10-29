"""Pipeline tests: malformed output, retries, validation, and API errors."""

from __future__ import annotations

import json
from typing import Any

import pytest

from exceptions import (
    EmptyResponseError,
    InvalidJSONError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    SchemaValidationError,
)
from extractor import Extractor, parse_json_object
from schemas import InvoiceExtraction, ResumeExtraction


def _extractor(
    settings: Any, stub_client_cls: type, responses: list[str | Exception]
) -> tuple[Extractor, Any]:
    client = stub_client_cls(responses)
    return Extractor(settings, client=client), client


def test_resume_extraction_succeeds(
    settings: Any, stub_client_cls: type, resume_json: str
) -> None:
    extractor, client = _extractor(settings, stub_client_cls, [resume_json])

    result = extractor.extract("resume", "Jordan Rivera, backend engineer")

    assert isinstance(result.data, ResumeExtraction)
    assert result.data.full_name == "Jordan A. Rivera"
    assert result.provider == "gemini"
    assert result.attempts == 1
    assert len(client.calls) == 1
    assert client.calls[0]["response_model"] is ResumeExtraction
    assert "Jordan Rivera, backend engineer" in client.calls[0]["user_prompt"]


def test_invoice_extraction_succeeds(
    settings: Any, stub_client_cls: type, invoice_json: str
) -> None:
    extractor, client = _extractor(settings, stub_client_cls, [invoice_json])

    result = extractor.extract("invoice", "Invoice EOS-2025-0042, total 323.88 GBP")

    assert isinstance(result.data, InvoiceExtraction)
    assert result.data.currency == "GBP"
    assert str(result.data.total_amount) == "323.88"
    assert client.calls[0]["response_model"] is InvoiceExtraction
    assert "invoice:" in result.summary()
    assert "323.88" in result.summary()
    assert "EOS-2025-0042" not in result.summary()


def test_code_fenced_json_is_accepted(
    settings: Any, stub_client_cls: type, resume_json: str
) -> None:
    fenced = f"```json\n{resume_json}\n```"
    extractor, _ = _extractor(settings, stub_client_cls, [fenced])

    result = extractor.extract("resume", "Resume text")

    assert result.data.full_name == "Jordan A. Rivera"


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "not json at all", '{"full_name": ', '["a list"]', "null"],
)
def test_empty_or_malformed_responses_are_rejected(raw: str) -> None:
    with pytest.raises((EmptyResponseError, InvalidJSONError)):
        parse_json_object(raw)


def test_nan_is_rejected() -> None:
    with pytest.raises(InvalidJSONError, match="non-finite"):
        parse_json_object('{"total_amount": NaN}')


def test_malformed_json_is_retried_once(
    settings: Any, stub_client_cls: type, resume_json: str
) -> None:
    extractor, client = _extractor(
        settings, stub_client_cls, ["this is not json", resume_json]
    )

    result = extractor.extract("resume", "Resume text")

    assert result.attempts == 2
    assert len(client.calls) == 2
    assert "CORRECTION REQUIRED" in client.calls[1]["user_prompt"]
    assert "CORRECTION REQUIRED" not in client.calls[0]["user_prompt"]


def test_invalid_schema_is_retried_then_fails(
    settings: Any, stub_client_cls: type
) -> None:
    bad = json.dumps(
        {
            "invoice_number": "X",
            "invoice_date": "14/03/2025",
            "due_date": None,
            "currency": "GBP",
            "supplier_name": None,
            "supplier_address": None,
            "customer_name": None,
            "subtotal": None,
            "tax_amount": None,
            "total_amount": "not a number",
            "line_items": [],
            "payment_terms": None,
        }
    )
    extractor, client = _extractor(settings, stub_client_cls, [bad, bad])

    with pytest.raises(SchemaValidationError) as excinfo:
        extractor.extract("invoice", "Invoice text")

    assert excinfo.value.attempts == 2
    assert len(client.calls) == 2
    assert "total_amount" in str(excinfo.value)
    assert "invoice_date" in str(excinfo.value)


def test_all_null_response_is_valid_without_retry(
    settings: Any, stub_client_cls: type
) -> None:
    all_null = json.dumps(
        {
            "full_name": None,
            "email": None,
            "phone": None,
            "professional_summary": None,
            "education": [],
            "technical_skills": [],
            "soft_skills": [],
            "work_experience": [],
            "projects": [],
            "certifications": [],
            "languages": [],
        }
    )
    extractor, client = _extractor(settings, stub_client_cls, [all_null])

    result = extractor.extract("resume", "Unrelated text with no CV details")

    assert result.attempts == 1
    assert result.data.full_name is None
    assert result.data.education == []
    assert len(client.calls) == 1


def test_zero_retries_makes_one_attempt(
    stub_client_cls: type, resume_json: str
) -> None:
    from config import Settings

    strict_settings = Settings(
        provider="openai",
        model="test-model",
        timeout_seconds=5.0,
        max_output_retries=0,
        api_key="test-key",
    )
    extractor, client = _extractor(
        strict_settings, stub_client_cls, ["not json"]
    )

    with pytest.raises(InvalidJSONError) as excinfo:
        extractor.extract("resume", "Resume text")

    assert excinfo.value.attempts == 1
    assert len(client.calls) == 1


def test_schema_error_is_not_retried_when_retries_disabled(
    stub_client_cls: type, invoice_payload: dict[str, Any]
) -> None:
    from config import Settings

    bad = dict(invoice_payload)
    bad["currency"] = "POUNDS"
    strict_settings = Settings(
        provider="gemini",
        model="test-model",
        timeout_seconds=5.0,
        max_output_retries=0,
        api_key="test-key",
    )
    extractor, client = _extractor(
        strict_settings, stub_client_cls, [json.dumps(bad)]
    )

    with pytest.raises(SchemaValidationError):
        extractor.extract("invoice", "Invoice text")
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "error",
    [
        LLMAuthenticationError("auth failed"),
        LLMRateLimitError("rate limited"),
        LLMTimeoutError("timed out"),
    ],
)
def test_provider_errors_are_not_retried(
    settings: Any, stub_client_cls: type, error: Exception
) -> None:
    extractor, client = _extractor(settings, stub_client_cls, [error])

    with pytest.raises(type(error)):
        extractor.extract("resume", "Resume text")

    assert len(client.calls) == 1


def test_empty_input_is_rejected_before_api_call(
    settings: Any, stub_client_cls: type, resume_json: str
) -> None:
    from exceptions import InputError

    extractor, client = _extractor(settings, stub_client_cls, [resume_json])

    with pytest.raises(InputError):
        extractor.extract("resume", "   ")

    assert client.calls == []


def test_invalid_mode_is_rejected(
    settings: Any, stub_client_cls: type, resume_json: str
) -> None:
    extractor, client = _extractor(settings, stub_client_cls, [resume_json])

    with pytest.raises(ValueError):
        extractor.extract("receipt", "Some text")

    assert client.calls == []


def test_parse_json_object_returns_dict() -> None:
    payload = parse_json_object('{"a": 1}')
    assert payload == {"a": 1}
