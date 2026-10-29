"""Shared deterministic fixtures. No test uses a real API or network."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Settings  # noqa: E402
from schemas import InvoiceExtraction, ResumeExtraction  # noqa: E402


@pytest.fixture
def settings() -> Settings:
    return Settings(
        provider="gemini",
        model="test-model",
        timeout_seconds=10.0,
        max_output_retries=1,
        api_key="test-key-not-real",
    )


@pytest.fixture
def resume_payload() -> dict[str, Any]:
    return {
        "full_name": "Jordan A. Rivera",
        "email": "jordan.rivera@example.com",
        "phone": "+1 555 014 7788",
        "professional_summary": "Backend engineer focused on payments.",
        "education": [
            {
                "institution": "University of Texas at Austin",
                "degree": "Bachelor of Science",
                "field_of_study": "Computer Science",
                "start_date": "Aug 2015",
                "end_date": "May 2019",
                "location": "Austin, TX",
                "grade": "3.7/4.0",
            }
        ],
        "technical_skills": ["Python", "Go", "PostgreSQL"],
        "soft_skills": ["Mentoring"],
        "work_experience": [
            {
                "job_title": "Senior Software Engineer",
                "company": "Northstar Payments",
                "location": "Austin, TX",
                "start_date": "Jan 2022",
                "end_date": None,
                "description": None,
                "highlights": ["Reduced p95 latency by 38%."],
            }
        ],
        "projects": [
            {
                "name": "Ledger Lens",
                "description": "Open-source reconciliation tool.",
                "technologies": ["Python", "FastAPI"],
                "url": "https://github.com/example/ledger-lens",
                "highlights": [],
            }
        ],
        "certifications": ["AWS Certified Solutions Architect - Associate"],
        "languages": [{"language": "English", "proficiency": "Native"}],
    }


@pytest.fixture
def invoice_payload() -> dict[str, Any]:
    return {
        "invoice_number": "EOS-2025-0042",
        "invoice_date": "2025-03-14",
        "due_date": "2025-04-13",
        "currency": "GBP",
        "supplier_name": "Evergreen Office Supplies Ltd",
        "supplier_address": "Unit 4, Bristol BS1 4XY, United Kingdom",
        "customer_name": "Harbourview Design Studio",
        "subtotal": 269.90,
        "tax_amount": 53.98,
        "total_amount": 323.88,
        "line_items": [
            {
                "description": "Recycled A4 paper",
                "quantity": 4,
                "unit_price": 24.50,
                "line_total": 98.00,
            }
        ],
        "payment_terms": "Net 30",
    }


@pytest.fixture
def resume_json(resume_payload: dict[str, Any]) -> str:
    return json.dumps(ResumeExtraction.model_validate(resume_payload).model_dump(mode="json"))


@pytest.fixture
def invoice_json(invoice_payload: dict[str, Any]) -> str:
    return json.dumps(InvoiceExtraction.model_validate(invoice_payload).model_dump(mode="json"))


class StubClient:
    """Deterministic stand-in for a provider client."""

    provider = "gemini"
    model = "test-model"

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_model": response_model,
            }
        )
        if not self._responses:
            raise AssertionError("StubClient received more calls than expected")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def stub_client_cls() -> type[StubClient]:
    return StubClient
