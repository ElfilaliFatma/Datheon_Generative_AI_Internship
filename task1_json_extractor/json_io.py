"""UTF-8 document input and validated JSON output helpers."""

from __future__ import annotations

import json
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any

from exceptions import InputError, OutputWriteError

MAX_INPUT_BYTES = 2_000_000
MAX_DOCUMENT_CHARS = 200_000


def validate_document_text(text: str) -> str:
    """Validate an in-memory document and return it unchanged."""
    if not isinstance(text, str):
        raise InputError("Document text must be a string.")
    if not text.strip():
        raise InputError("The document is empty. Provide resume or invoice text.")
    if "\x00" in text:
        raise InputError("The document contains NUL characters and is not plain text.")
    if len(text) > MAX_DOCUMENT_CHARS:
        raise InputError(
            f"The document is too large ({len(text)} characters). "
            f"The limit is {MAX_DOCUMENT_CHARS} characters."
        )
    return text


def read_document_file(path: Path | str) -> str:
    """Read a UTF-8 (or UTF-8 with BOM) text document from disk."""
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError:
        raise InputError(f"Input file does not exist: {path}") from None
    except OSError as exc:
        raise InputError(f"Could not access input file: {path}") from exc

    if not resolved.is_file():
        raise InputError(f"Input path is not a regular file: {resolved}")

    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise InputError(f"Could not inspect input file: {resolved}") from exc
    if size > MAX_INPUT_BYTES:
        raise InputError(
            f"Input file is too large ({size} bytes). "
            f"The limit is {MAX_INPUT_BYTES} bytes."
        )

    try:
        raw = resolved.read_bytes()
    except (OSError, PermissionError) as exc:
        raise InputError(f"Could not read input file: {resolved}") from exc

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise InputError(
            f"Input file is not valid UTF-8: {resolved}. "
            "Save it as UTF-8 and try again."
        ) from None

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return validate_document_text(text)


def serialize_model(model: object) -> str:
    """Serialize a Pydantic model as readable, validated JSON text.

    Decimal values are written as exact JSON numbers rather than being routed
    through binary floating point, so currency precision is preserved.
    """
    dump = getattr(model, "model_dump", None)
    if not callable(dump):
        raise OutputWriteError("The result is not a valid Pydantic model.")
    try:
        data = dump(mode="python")
        text = dumps_json(data)
        # Verify the exact text can be parsed again before returning it.
        json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OutputWriteError(
            "The extraction result could not be serialized as valid JSON."
        ) from exc
    return text + "\n"


def dumps_json(value: Any, *, indent: int = 2) -> str:
    """Encode JSON with exact Decimal numbers and readable indentation."""
    return _encode_json(value, level=0, indent=indent)


def _encode_json(value: Any, *, level: int, indent: int) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite Decimal values are not valid JSON")
        if value == 0:
            return "0"
        text = format(value, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        if text in {"", "-0"}:
            return "0"
        return text
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("non-finite float values are not valid JSON")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        if not value:
            return "{}"
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        padding = " " * (indent * (level + 1))
        closing = " " * (indent * level)
        entries = [
            f"{padding}{json.dumps(key, ensure_ascii=False)}: "
            f"{_encode_json(item, level=level + 1, indent=indent)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(entries) + f"\n{closing}}}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        padding = " " * (indent * (level + 1))
        closing = " " * (indent * level)
        entries = [
            padding + _encode_json(item, level=level + 1, indent=indent)
            for item in value
        ]
        return "[\n" + ",\n".join(entries) + f"\n{closing}]"
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


def write_json_file(path: Path | str, json_text: str) -> Path:
    """Atomically write validated JSON text as UTF-8 with a trailing newline."""
    try:
        json.loads(json_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise OutputWriteError("Refusing to write invalid JSON.") from exc

    destination = Path(path).expanduser()
    try:
        parent = destination.parent.resolve(strict=False)
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputWriteError(
            f"Could not create the output directory: {destination.parent}"
        ) from exc

    if destination.exists() and destination.is_dir():
        raise OutputWriteError(f"Output path is a directory: {destination}")

    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=parent,
            prefix=".extraction-",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_name = temp_file.name
            temp_file.write(json_text)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, destination)
    except OSError as exc:
        if temp_name:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise OutputWriteError(
            f"Could not write the output JSON file: {destination}"
        ) from exc

    return destination.resolve()
