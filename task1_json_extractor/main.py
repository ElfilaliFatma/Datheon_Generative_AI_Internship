"""Command-line entry point for the structured LLM extraction pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from config import Settings
from exceptions import (
    ApplicationError,
    ConfigurationError,
    ExtractionError,
    InputError,
    OutputWriteError,
)
from extractor import Extractor, ExtractionResult
from json_io import read_document_file, write_json_file
from schemas import ExtractionMode

EXIT_SUCCESS = 0
EXIT_USAGE = 2
EXIT_CONFIGURATION = 3
EXIT_INPUT = 4
EXIT_EXTRACTION = 5
EXIT_OUTPUT = 6


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Extract validated structured JSON from resume or invoice text "
            "using the LLM provider configured in LLM_PROVIDER."
        ),
        epilog=(
            "Examples:\n"
            "  python main.py --mode resume --input examples/sample_resume.txt\n"
            "  python main.py --mode invoice --input examples/sample_invoice.txt "
            "--output output/invoice.json\n"
            "  python main.py --mode resume\n"
            "\n"
            "The selected document text is sent to the configured external "
            "LLM API. API usage may incur charges."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        required=False,
        choices=[mode.value for mode in ExtractionMode],
        help="Extraction mode: resume or invoice (required unless prompted).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        metavar="PATH",
        help=(
            "UTF-8 text file containing the document. If omitted, "
            "interactive paste mode starts."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="PATH",
        help="Optional path for the validated JSON result.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        mode = _resolve_mode(args.mode)
        document = (
            read_document_file(args.input)
            if args.input is not None
            else _read_interactive_document(mode)
        )

        settings = Settings.from_env()
        print(
            f"Sending document text to the configured {settings.provider_label} "
            f"API (model: {settings.model}). API usage may incur charges.",
            file=sys.stderr,
        )
        extractor = Extractor(settings)
        result = extractor.extract(mode, document)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        print(
            "Copy .env.example to .env and set LLM_PROVIDER plus the matching "
            "API key. Never commit .env.",
            file=sys.stderr,
        )
        return EXIT_CONFIGURATION
    except InputError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except ExtractionError as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        if exc.attempts > 1:
            print(f"Attempts made: {exc.attempts}", file=sys.stderr)
        print(
            "No result was saved. Check the configuration and provider status, "
            "then try again.",
            file=sys.stderr,
        )
        return EXIT_EXTRACTION
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except ApplicationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    _print_result(result, include_json=args.output is None)

    if args.output is not None:
        try:
            saved_path = write_json_file(args.output, result.to_json())
        except OutputWriteError as exc:
            print(f"Output error: {exc}", file=sys.stderr)
            return EXIT_OUTPUT
        print(f"Saved validated JSON to: {saved_path}")

    return EXIT_SUCCESS


def _resolve_mode(raw_mode: str | None) -> ExtractionMode:
    if raw_mode:
        return ExtractionMode(raw_mode)
    if not sys.stdin.isatty():
        raise InputError(
            "No --mode supplied and stdin is not interactive. "
            "Pass --mode resume or --mode invoice."
        )
    print("Select extraction mode:")
    print("  1) resume")
    print("  2) invoice")
    try:
        choice = input("Enter 1 or 2: ").strip().lower()
    except EOFError:
        raise InputError("No extraction mode supplied.") from None
    mapping = {
        "1": ExtractionMode.RESUME,
        "2": ExtractionMode.INVOICE,
        "resume": ExtractionMode.RESUME,
        "invoice": ExtractionMode.INVOICE,
    }
    if choice not in mapping:
        raise InputError("Invalid mode choice. Use 1 (resume) or 2 (invoice).")
    return mapping[choice]


def _read_interactive_document(mode: ExtractionMode) -> str:
    if not sys.stdin.isatty():
        raise InputError(
            "No --input supplied and stdin is not interactive. "
            "Pass --input PATH or run in a terminal to paste text."
        )
    print()
    print(f"Paste the {mode.value} text below.")
    print("Finish with a line containing only a single dot (.) and press Enter.")
    print(
        "The text you paste will be sent to the configured external LLM API.",
        file=sys.stderr,
    )
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == ".":
            break
        lines.append(line)
    document = "\n".join(lines)
    if not document.strip():
        raise InputError("No document text was entered.")
    return document


def _print_result(result: ExtractionResult, *, include_json: bool) -> None:
    print()
    print(
        f"Extraction succeeded ({result.provider}, model={result.model}, "
        f"attempts={result.attempts})."
    )
    print(f"Summary: {result.summary()}")
    if include_json:
        print()
        print(result.to_json(), end="")


if __name__ == "__main__":
    raise SystemExit(main())
