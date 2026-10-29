"""JSON serialization, file I/O, and CLI behaviour tests."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from exceptions import InputError, OutputWriteError
from extractor import Extractor
from json_io import (
    read_document_file,
    serialize_model,
    validate_document_text,
    write_json_file,
)
from main import main
from schemas import InvoiceExtraction, ResumeExtraction


def test_serialized_resume_is_valid_json_with_null_and_lists(
    resume_payload: dict[str, Any],
) -> None:
    resume = ResumeExtraction.model_validate(resume_payload)

    text = serialize_model(resume)
    parsed = json.loads(text)

    assert parsed["full_name"] == "Jordan A. Rivera"
    assert parsed["education"][0]["institution"] == "University of Texas at Austin"
    assert parsed["work_experience"][0]["end_date"] is None
    assert text.endswith("\n")
    assert "\\u" not in text


def test_unicode_is_preserved_in_json_output(
    resume_payload: dict[str, Any],
) -> None:
    resume_payload["full_name"] = "José Álvarez"
    text = serialize_model(ResumeExtraction.model_validate(resume_payload))

    assert "José Álvarez" in text
    assert json.loads(text)["full_name"] == "José Álvarez"


def test_invoice_amounts_serialize_as_json_numbers(
    invoice_payload: dict[str, Any],
) -> None:
    invoice = InvoiceExtraction.model_validate(invoice_payload)
    parsed = json.loads(serialize_model(invoice))

    assert parsed["total_amount"] == 323.88
    assert isinstance(parsed["total_amount"], float)
    assert isinstance(parsed["line_items"][0]["unit_price"], float)
    assert not isinstance(parsed["total_amount"], str)


def test_large_decimal_amounts_keep_exact_precision() -> None:
    invoice = InvoiceExtraction(
        subtotal=Decimal("0.10"),
        tax_amount=Decimal("0.02"),
        total_amount=Decimal("12345678901234.5678"),
    )

    text = serialize_model(invoice)
    parsed = json.loads(text, parse_float=Decimal)

    assert '"subtotal": 0.1,' in text
    assert parsed["total_amount"] == Decimal("12345678901234.5678")
    assert parsed["total_amount"] != Decimal("12345678901234.568")


def test_expected_example_files_match_schema() -> None:
    root = Path(__file__).resolve().parents[1]
    resume_example = json.loads(
        (root / "examples" / "expected_resume_schema.json").read_text(
            encoding="utf-8"
        )
    )
    invoice_example = json.loads(
        (root / "examples" / "expected_invoice_schema.json").read_text(
            encoding="utf-8"
        )
    )
    resume_example.pop("$comment", None)
    invoice_example.pop("$comment", None)

    ResumeExtraction.model_validate(resume_example)
    InvoiceExtraction.model_validate(invoice_example)


def test_read_utf8_file_with_bom(tmp_path: Path) -> None:
    path = tmp_path / "resume.txt"
    path.write_bytes("\ufeffJane Doe\nPython".encode("utf-8"))

    text = read_document_file(path)

    assert text.startswith("Jane Doe")
    assert "Python" in text


def test_read_missing_file_raises_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="does not exist"):
        read_document_file(tmp_path / "missing.txt")


def test_read_directory_raises_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="not a regular file"):
        read_document_file(tmp_path)


def test_non_utf8_file_raises_input_error(tmp_path: Path) -> None:
    path = tmp_path / "latin1.txt"
    path.write_bytes("Résumé".encode("latin-1"))

    with pytest.raises(InputError, match="not valid UTF-8"):
        read_document_file(path)


def test_empty_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")

    with pytest.raises(InputError, match="empty"):
        read_document_file(path)


def test_write_json_file_round_trips(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "result.json"
    text = '{"name": "José", "amount": 1.5}\n'

    saved = write_json_file(destination, text)

    assert saved == destination.resolve()
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "name": "José",
        "amount": 1.5,
    }


def test_write_json_file_rejects_invalid_json(tmp_path: Path) -> None:
    with pytest.raises(OutputWriteError, match="invalid JSON"):
        write_json_file(tmp_path / "bad.json", "{invalid")

    assert not (tmp_path / "bad.json").exists()


class _FakeClient:
    provider = "gemini"
    model = "test-model"

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type,
    ) -> str:
        return self._responses.pop(0)


def test_extraction_result_to_json_round_trips(
    settings: Any, invoice_json: str
) -> None:
    result = Extractor(settings, client=_FakeClient([invoice_json])).extract(
        "invoice", "Invoice total 323.88 GBP"
    )

    text = result.to_json()
    parsed = json.loads(text)

    assert parsed["invoice_number"] == "EOS-2025-0042"
    assert parsed["total_amount"] == 323.88
    assert parsed["line_items"][0]["description"] == "Recycled A4 paper"


def test_cli_help_displays(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "--mode" in output
    assert "--input" in output
    assert "--output" in output
    assert "resume" in output
    assert "invoice" in output
    assert "external LLM API" in output


def test_cli_requires_configuration_for_valid_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "resume.txt"
    path.write_text("Jane Doe\nPython", encoding="utf-8")
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code = main(["--mode", "resume", "--input", str(path)])

    assert exit_code == 3
    assert "Configuration error" in capsys.readouterr().err


def test_cli_handles_missing_input_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        ["--mode", "invoice", "--input", str(tmp_path / "missing.txt")]
    )

    assert exit_code == 4
    assert "Input error" in capsys.readouterr().err


def test_cli_saves_validated_json_without_printing_document_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    invoice_json: str,
) -> None:
    from config import Settings
    import main as main_module

    input_path = tmp_path / "invoice.txt"
    output_path = tmp_path / "result.json"
    input_path.write_text("Invoice EOS-2025-0042 total 323.88 GBP", encoding="utf-8")

    fake_settings = Settings(
        provider="gemini",
        model="test-model",
        timeout_seconds=10.0,
        max_output_retries=1,
        api_key="test-key",
    )

    class _Client:
        provider = "gemini"
        model = "test-model"

        def generate_json(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            response_model: type,
        ) -> str:
            return invoice_json

    monkeypatch.setattr(
        main_module.Settings, "from_env", staticmethod(lambda: fake_settings)
    )
    monkeypatch.setattr(
        main_module, "Extractor", lambda settings: Extractor(settings, client=_Client())
    )

    exit_code = main_module.main(
        [
            "--mode",
            "invoice",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "EOS-2025-0042" not in captured.out
    assert "invoice: line_items=1" in captured.out
    assert "Saved validated JSON to:" in captured.out
    assert json.loads(output_path.read_text(encoding="utf-8"))[
        "invoice_number"
    ] == "EOS-2025-0042"


def test_validate_document_text_rejects_oversized_document() -> None:
    with pytest.raises(InputError, match="too large"):
        validate_document_text("a" * 200_001)


def test_validate_document_text_accepts_normal_documents() -> None:
    text = "Jane Doe\nPython developer"
    assert validate_document_text(text) == text
