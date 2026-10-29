"""Pydantic schemas and validation rules for the two extraction modes."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from email_validator import EmailNotValidError, validate_email
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    WithJsonSchema,
    field_validator,
    model_validator,
)


class ExtractionMode(str, Enum):
    """Supported document extraction modes."""

    RESUME = "resume"
    INVOICE = "invoice"


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _reject_boolean_number(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("must be a number, not a boolean")
    return value


def _validate_amount(value: Decimal) -> Decimal:
    if not value.is_finite():
        raise ValueError("must be a finite number")
    if value < 0:
        raise ValueError("must be zero or greater")
    if value >= Decimal("1e18"):
        raise ValueError("must be less than 1e18")
    if len(value.as_tuple().digits) > 18:
        raise ValueError("has too many digits")
    if -value.as_tuple().exponent > 4:
        raise ValueError("must have at most 4 decimal places")
    return value


def _validate_quantity(value: Decimal) -> Decimal:
    if not value.is_finite():
        raise ValueError("must be a finite number")
    if value < 0:
        raise ValueError("must be zero or greater")
    if value >= Decimal("1e12"):
        raise ValueError("must be less than 1e12")
    if -value.as_tuple().exponent > 3:
        raise ValueError("must have at most 3 decimal places")
    return value


def _decimal_to_number(value: Decimal) -> float:
    """Serialize Decimal values as JSON numbers (not JSON strings)."""
    return float(value)


Amount = Annotated[
    Decimal,
    BeforeValidator(_reject_boolean_number),
    AfterValidator(_validate_amount),
    PlainSerializer(_decimal_to_number, return_type=float, when_used="json"),
    WithJsonSchema(
        {
            "type": "number",
            "description": "Non-negative decimal amount; use a JSON number.",
        }
    ),
]
Quantity = Annotated[
    Decimal,
    BeforeValidator(_reject_boolean_number),
    AfterValidator(_validate_quantity),
    PlainSerializer(_decimal_to_number, return_type=float, when_used="json"),
    WithJsonSchema(
        {
            "type": "number",
            "description": "Non-negative decimal quantity; use a JSON number.",
        }
    ),
]
OptionalText = Annotated[str | None, BeforeValidator(_blank_to_none)]


def _normalise_string_list(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            # Let Pydantic report the type error for this element.
            return value
        text = item.strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


StringList = Annotated[list[str], BeforeValidator(_normalise_string_list)]


class StrictModel(BaseModel):
    """Base model that rejects unknown keys and strips surrounding whitespace."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class EducationEntry(StrictModel):
    """One education record."""

    institution: OptionalText = Field(
        default=None, description="School, college, or university name."
    )
    degree: OptionalText = Field(
        default=None, description="Degree or qualification, for example BSc."
    )
    field_of_study: OptionalText = Field(
        default=None, description="Subject or major, if stated."
    )
    start_date: OptionalText = Field(
        default=None, description="Start date exactly as stated or normalized."
    )
    end_date: OptionalText = Field(
        default=None, description="End date or expected graduation date."
    )
    location: OptionalText = Field(default=None, description="Location, if stated.")
    grade: OptionalText = Field(
        default=None, description="Grade, GPA, or classification, if stated."
    )

    @model_validator(mode="after")
    def _must_contain_information(self) -> "EducationEntry":
        if not any(
            value not in (None, "", []) for value in self.model_dump().values()
        ):
            raise ValueError("education entry must contain at least one value")
        return self


class WorkExperienceEntry(StrictModel):
    """One employment record."""

    job_title: OptionalText = Field(default=None, description="Role or job title.")
    company: OptionalText = Field(default=None, description="Employer name.")
    location: OptionalText = Field(default=None, description="Work location.")
    start_date: OptionalText = Field(default=None, description="Start date.")
    end_date: OptionalText = Field(
        default=None, description="End date, or null if current."
    )
    description: OptionalText = Field(
        default=None, description="Short responsibility summary stated in the document."
    )
    highlights: StringList = Field(
        default_factory=list, description="Specific achievements or duties."
    )

    @model_validator(mode="after")
    def _must_contain_information(self) -> "WorkExperienceEntry":
        if not any(
            value not in (None, "", []) for value in self.model_dump().values()
        ):
            raise ValueError("work experience entry must contain at least one value")
        return self


class ProjectEntry(StrictModel):
    """One project record."""

    name: OptionalText = Field(default=None, description="Project name.")
    description: OptionalText = Field(
        default=None, description="Project description stated in the document."
    )
    technologies: StringList = Field(
        default_factory=list, description="Technologies explicitly named."
    )
    url: OptionalText = Field(default=None, description="Project URL, if stated.")
    highlights: StringList = Field(
        default_factory=list, description="Notable outcomes or responsibilities."
    )

    @model_validator(mode="after")
    def _must_contain_information(self) -> "ProjectEntry":
        if not any(
            value not in (None, "", []) for value in self.model_dump().values()
        ):
            raise ValueError("project entry must contain at least one value")
        return self


class LanguageEntry(StrictModel):
    """One language and, when stated, its proficiency."""

    language: str = Field(description="Language name.")
    proficiency: OptionalText = Field(
        default=None, description="Proficiency level exactly as stated."
    )

    @field_validator("language")
    @classmethod
    def _language_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class ResumeExtraction(StrictModel):
    """Structured information extracted from a CV or resume."""

    full_name: OptionalText = Field(default=None, description="Candidate full name.")
    email: OptionalText = Field(default=None, description="Email address.")
    phone: OptionalText = Field(default=None, description="Phone number.")
    professional_summary: OptionalText = Field(
        default=None, description="Professional summary or objective."
    )
    education: list[EducationEntry] = Field(
        default_factory=list, description="Education entries."
    )
    technical_skills: StringList = Field(
        default_factory=list, description="Technical skills."
    )
    soft_skills: StringList = Field(
        default_factory=list, description="Soft skills."
    )
    work_experience: list[WorkExperienceEntry] = Field(
        default_factory=list, description="Work experience entries."
    )
    projects: list[ProjectEntry] = Field(
        default_factory=list, description="Project entries."
    )
    certifications: StringList = Field(
        default_factory=list, description="Certifications."
    )
    languages: list[LanguageEntry] = Field(
        default_factory=list, description="Language entries."
    )

    @field_validator("email")
    @classmethod
    def _valid_email_or_null(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            result = validate_email(value, check_deliverability=False)
        except EmailNotValidError:
            raise ValueError("must be a valid email address or null") from None
        return result.normalized


class InvoiceLineItem(StrictModel):
    """One invoice line item."""

    description: str = Field(description="Line item description.")
    quantity: Quantity | None = Field(default=None, description="Quantity.")
    unit_price: Amount | None = Field(default=None, description="Price per unit.")
    line_total: Amount | None = Field(default=None, description="Line total.")

    @field_validator("description")
    @classmethod
    def _description_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class InvoiceExtraction(StrictModel):
    """Structured information extracted from an invoice."""

    invoice_number: OptionalText = Field(default=None, description="Invoice number.")
    invoice_date: OptionalText = Field(
        default=None, description="Invoice date as ISO 8601 YYYY-MM-DD."
    )
    due_date: OptionalText = Field(
        default=None, description="Payment due date as ISO 8601 YYYY-MM-DD."
    )
    currency: OptionalText = Field(
        default=None, description="Three-letter ISO 4217 currency code."
    )
    supplier_name: OptionalText = Field(default=None, description="Supplier name.")
    supplier_address: OptionalText = Field(
        default=None, description="Supplier address."
    )
    customer_name: OptionalText = Field(default=None, description="Customer name.")
    subtotal: Amount | None = Field(default=None, description="Subtotal amount.")
    tax_amount: Amount | None = Field(default=None, description="Tax amount.")
    total_amount: Amount | None = Field(default=None, description="Total amount.")
    line_items: list[InvoiceLineItem] = Field(
        default_factory=list, description="Invoice line item entries."
    )
    payment_terms: OptionalText = Field(
        default=None, description="Payment terms stated on the invoice."
    )

    @field_validator("invoice_date", "due_date")
    @classmethod
    def _valid_iso_date_or_null(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _ISO_DATE_RE.fullmatch(value):
            raise ValueError("must use ISO 8601 format YYYY-MM-DD or be null")
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            raise ValueError("must be a real calendar date or null") from None
        return parsed.isoformat()

    @field_validator("currency")
    @classmethod
    def _valid_currency_or_null(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", normalized):
            raise ValueError("must be a three-letter ISO 4217 code or null")
        return normalized


SCHEMA_BY_MODE: dict[ExtractionMode, type[StrictModel]] = {
    ExtractionMode.RESUME: ResumeExtraction,
    ExtractionMode.INVOICE: InvoiceExtraction,
}


def schema_for_mode(mode: ExtractionMode | str) -> type[StrictModel]:
    """Return the Pydantic schema for an extraction mode."""
    try:
        normalized = ExtractionMode(mode)
    except ValueError:
        raise ValueError("mode must be 'resume' or 'invoice'") from None
    return SCHEMA_BY_MODE[normalized]


def build_provider_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Build a strict JSON Schema accepted by both providers.

    Every object gets ``additionalProperties: false`` and lists all properties
    as required. Optionality is expressed through ``anyOf`` including null,
    which is how Pydantic represents nullable fields. UI-only keywords and
    defaults are removed to stay within the providers' supported subset.
    """
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise TypeError("model must be a Pydantic BaseModel subclass")
    schema = model.model_json_schema()
    _normalise_schema_node(schema)
    return schema


def _normalise_schema_node(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _normalise_schema_node(item)
        return
    if not isinstance(node, dict):
        return

    node.pop("default", None)
    node.pop("examples", None)
    node.pop("example", None)
    node.pop("title", None)
    node.pop("discriminator", None)

    properties = node.get("properties")
    if isinstance(properties, dict):
        node["type"] = "object"
        node["additionalProperties"] = False
        node["required"] = list(properties)

    for key, value in list(node.items()):
        if key in {"properties", "$defs", "definitions"} and isinstance(
            value, dict
        ):
            for child in value.values():
                _normalise_schema_node(child)
        else:
            _normalise_schema_node(value)
