"""Schema validation tests for resumes and invoices."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from schemas import (
    EducationEntry,
    ExtractionMode,
    InvoiceExtraction,
    InvoiceLineItem,
    ResumeExtraction,
    WorkExperienceEntry,
    build_provider_json_schema,
    schema_for_mode,
)


def test_valid_resume_payload(resume_payload: dict[str, Any]) -> None:
    resume = ResumeExtraction.model_validate(resume_payload)

    assert resume.full_name == "Jordan A. Rivera"
    assert resume.email == "jordan.rivera@example.com"
    assert len(resume.education) == 1
    assert resume.education[0].institution == "University of Texas at Austin"
    assert resume.work_experience[0].end_date is None
    assert resume.technical_skills == ["Python", "Go", "PostgreSQL"]
    assert resume.languages[0].proficiency == "Native"


def test_valid_invoice_payload(invoice_payload: dict[str, Any]) -> None:
    invoice = InvoiceExtraction.model_validate(invoice_payload)

    assert invoice.invoice_number == "EOS-2025-0042"
    assert invoice.total_amount == Decimal("323.88")
    assert invoice.currency == "GBP"
    assert invoice.line_items[0].unit_price == Decimal("24.50")
    assert isinstance(invoice.line_items[0].quantity, Decimal)


def test_numeric_strings_are_coerced_to_decimal(
    invoice_payload: dict[str, Any],
) -> None:
    invoice_payload["subtotal"] = "269.90"
    invoice_payload["line_items"][0]["quantity"] = "4"

    invoice = InvoiceExtraction.model_validate(invoice_payload)

    assert invoice.subtotal == Decimal("269.90")
    assert invoice.line_items[0].quantity == Decimal("4")


def test_missing_optional_values_become_null_and_empty_lists() -> None:
    resume = ResumeExtraction.model_validate({})
    invoice = InvoiceExtraction.model_validate({})

    assert resume.full_name is None
    assert resume.education == []
    assert resume.technical_skills == []
    assert resume.languages == []
    assert invoice.invoice_date is None
    assert invoice.total_amount is None
    assert invoice.line_items == []
    assert invoice.model_dump(mode="json")["line_items"] == []


def test_blank_strings_are_normalised_to_null(
    resume_payload: dict[str, Any],
) -> None:
    resume_payload["phone"] = "   "
    resume_payload["professional_summary"] = ""

    resume = ResumeExtraction.model_validate(resume_payload)

    assert resume.phone is None
    assert resume.professional_summary is None


def test_strings_are_stripped_and_duplicates_removed(
    resume_payload: dict[str, Any],
) -> None:
    resume_payload["full_name"] = "  Jordan Rivera  "
    resume_payload["technical_skills"] = ["Python", " python ", "", "SQL"]

    resume = ResumeExtraction.model_validate(resume_payload)

    assert resume.full_name == "Jordan Rivera"
    assert resume.technical_skills == ["Python", "SQL"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("full_name", 123),
        ("email", "not-an-email"),
        ("technical_skills", "Python"),
        ("education", {"institution": "MIT"}),
    ],
)
def test_invalid_resume_types_are_rejected(
    resume_payload: dict[str, Any], field: str, value: object
) -> None:
    resume_payload[field] = value

    with pytest.raises(ValidationError):
        ResumeExtraction.model_validate(resume_payload)


def test_unknown_fields_are_rejected(
    resume_payload: dict[str, Any],
) -> None:
    resume_payload["invented_field"] = "hallucination"

    with pytest.raises(ValidationError):
        ResumeExtraction.model_validate(resume_payload)


@pytest.mark.parametrize(
    "bad_date", ["14/03/2025", "2025-02-30", "2025-13-01", "March 14, 2025"]
)
def test_invalid_invoice_dates_are_rejected(
    invoice_payload: dict[str, Any], bad_date: str
) -> None:
    invoice_payload["invoice_date"] = bad_date

    with pytest.raises(ValidationError):
        InvoiceExtraction.model_validate(invoice_payload)


def test_valid_invoice_date_is_normalised(
    invoice_payload: dict[str, Any],
) -> None:
    invoice_payload["due_date"] = "2025-04-13"

    invoice = InvoiceExtraction.model_validate(invoice_payload)

    assert invoice.due_date == "2025-04-13"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_amount", "not-a-number"),
        ("total_amount", True),
        ("total_amount", -1),
        ("total_amount", "NaN"),
        ("total_amount", "Infinity"),
        ("total_amount", "1.23456"),
        ("total_amount", "1e18"),
        ("currency", "POUNDS"),
        ("currency", "G1P"),
    ],
)
def test_invalid_invoice_values_are_rejected(
    invoice_payload: dict[str, Any], field: str, value: object
) -> None:
    invoice_payload[field] = value

    with pytest.raises(ValidationError):
        InvoiceExtraction.model_validate(invoice_payload)


def test_currency_is_normalised_to_uppercase(
    invoice_payload: dict[str, Any],
) -> None:
    invoice_payload["currency"] = "gbp"

    invoice = InvoiceExtraction.model_validate(invoice_payload)

    assert invoice.currency == "GBP"


def test_line_item_rejects_blank_description() -> None:
    with pytest.raises(ValidationError):
        InvoiceLineItem.model_validate(
            {
                "description": " ",
                "quantity": None,
                "unit_price": None,
                "line_total": None,
            }
        )


def test_empty_nested_entries_are_rejected() -> None:
    with pytest.raises(ValidationError):
        EducationEntry.model_validate({})
    with pytest.raises(ValidationError):
        WorkExperienceEntry.model_validate({})


def test_schema_for_mode_returns_expected_models() -> None:
    assert schema_for_mode("resume") is ResumeExtraction
    assert schema_for_mode(ExtractionMode.INVOICE) is InvoiceExtraction
    with pytest.raises(ValueError):
        schema_for_mode("receipt")


def test_provider_schema_is_strict_for_objects() -> None:
    for model in (ResumeExtraction, InvoiceExtraction):
        schema = build_provider_json_schema(model)
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        assert "default" not in schema

    resume_schema = build_provider_json_schema(ResumeExtraction)
    education_schema = resume_schema["$defs"]["EducationEntry"]
    assert education_schema["additionalProperties"] is False
    assert set(education_schema["required"]) == set(
        education_schema["properties"]
    )


def test_resume_email_must_be_valid_or_null(
    resume_payload: dict[str, Any],
) -> None:
    resume_payload["email"] = None
    assert ResumeExtraction.model_validate(resume_payload).email is None

    resume_payload["email"] = "invalid@@example"
    with pytest.raises(ValidationError):
        ResumeExtraction.model_validate(resume_payload)
