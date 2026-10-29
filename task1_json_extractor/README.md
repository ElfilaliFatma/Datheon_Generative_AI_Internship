# Task 1 - Structured LLM Prompt Pipeline & JSON Extractor

A Python command-line application that turns unstructured resume or invoice text
into strictly validated JSON using Google Gemini or OpenAI.

> **Live extraction requires a configured LLM API key.** The document you supply
> is transmitted to the selected external provider. API usage may incur charges
> depending on the provider and your account.

## 1. Project overview

The application implements a small, auditable extraction pipeline:

```
input text -> mode -> prompt -> LLM -> JSON parse -> Pydantic validation
           -> strict JSON serialization -> terminal summary / JSON file
```

Nothing is fabricated when extraction fails. Provider errors, empty responses,
malformed JSON, and schema violations produce a clear non-zero exit code and no
output file. This project was built for the Datheon Python + Generative AI &
LangChain internship, Task 1. It does **not** claim Datheon approval, grading, or
a certificate.

## 2. Internship task objective

Build a complete, working Python application that extracts structured
information from unstructured text using an LLM API, produces strictly validated
JSON, and handles invalid responses and API errors safely, supporting CV/resume
and invoice documents.

## 3. Features

- Two extraction modes: `resume` and `invoice`.
- Provider selection with `LLM_PROVIDER=gemini` or `LLM_PROVIDER=openai`.
- Only the selected provider's API key is required.
- Native structured output:
  - Gemini: `response_mime_type="application/json"` with `response_json_schema`.
  - OpenAI: `response_format` with `type="json_schema"` and `strict=True`.
- Pydantic v2 schemas with nested models, email validation, ISO date checks,
  currency-code normalization, and non-negative decimal amounts.
- Prompt templates with two worked examples per mode, an explicit schema, and
  untrusted-document delimiters to resist prompt injection.
- Bounded retry (default: one extra attempt) for malformed or schema-invalid
  model output. Provider errors are never retried in a loop.
- Clear, typed errors with safe messages that exclude API keys and document text.
- UTF-8 file input, interactive paste input, terminal output, and JSON file output.
- Atomic JSON writes with a parse check before saving.
- 106 offline automated tests; no test uses a real API key or network call.
  Tests ignore any local `.env`, so they pass on a fresh clone and on a
  configured machine alike.

## 4. Architecture and data flow

| File | Responsibility |
| --- | --- |
| `main.py` | CLI parsing, interactive input, exit codes, terminal summary |
| `config.py` | Reads `LLM_PROVIDER`, the matching API key, model, timeout, retry limit |
| `schemas.py` | Pydantic models, validation rules, strict provider JSON Schema |
| `prompts.py` | Parameterized system/user prompt templates and few-shot examples |
| `llm_client.py` | Gemini and OpenAI clients plus safe error translation |
| `extractor.py` | Orchestration, JSON parsing, validation, bounded retries |
| `json_io.py` | UTF-8 document reading, exact Decimal JSON encoding, atomic writes |
| `exceptions.py` | Application error hierarchy |
| `tests/` | Schema, prompt, extractor, provider-client, config, and JSON tests |

Data flow:

1. `main.py` resolves the mode and reads a UTF-8 file or interactive paste.
2. `config.Settings.from_env()` validates provider, key, model, timeout, and retries.
3. `prompts.build_extraction_prompt()` embeds the document inside untrusted-data
   delimiters together with the expected JSON Schema and two examples.
4. `llm_client` calls the configured provider with native structured output.
5. `extractor.parse_json_object()` strips one optional code fence and parses JSON,
   rejecting `NaN`/`Infinity` and non-object responses.
6. Pydantic validates the payload strictly; unknown fields are rejected.
7. `json_io.serialize_model()` writes exact JSON numbers and verified UTF-8 text.
8. The CLI prints a PII-minimizing summary and saves the JSON only on success.

## 5. Technologies used

- Python 3.12 (3.10+ should work; not separately tested here)
- Pydantic 2.x and `email-validator`
- `google-genai` (official Google Gemini SDK)
- `openai` (official OpenAI SDK)
- `python-dotenv`
- pytest (tests only)

## 6. Prerequisites

- Python 3.10 or newer with `pip`.
- An API key for exactly one provider:
  - [Google AI Studio](https://aistudio.google.com/apikey) for Gemini.
  - [OpenAI Platform](https://platform.openai.com/api-keys) for OpenAI.
- A model available to your account that supports structured outputs.

## 7. Installation

### Windows PowerShell

```powershell
cd Datheon_Generative_AI_Internship\task1_json_extractor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, either run the interpreter directly without
activating (`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`) or
adjust the policy for the current process only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Linux / macOS

```bash
cd Datheon_Generative_AI_Internship/task1_json_extractor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

## 8. Virtual environment setup

The commands above create `.venv/` inside the project. `.venv/` is ignored by
Git. Deactivate with `deactivate`. Always install into the virtual environment
rather than globally.

## 9. API key configuration

Copy `.env.example` to `.env`, then edit it. Only the selected provider's key is
required.

Gemini:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
```

OpenAI:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

Optional settings:

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_TIMEOUT_SECONDS` | `60` | Request timeout, 5-300 seconds |
| `LLM_MAX_OUTPUT_RETRIES` | `1` | Extra attempts after bad output, 0-3 |

Environment variables already set in the shell take precedence over `.env`.
Keys are never printed. `Settings.__repr__` and `public_summary()` exclude the key.

## 10. Run CV/resume extraction

```powershell
python main.py --mode resume --input examples\sample_resume.txt
python main.py --mode resume --input examples\sample_resume.txt --output output\resume.json
python main.py --mode resume
```

The last command starts interactive paste mode: paste the resume, then enter a
line containing only `.` to finish and send it.

## 11. Run invoice extraction

```powershell
python main.py --mode invoice --input examples\sample_invoice.txt
python main.py --mode invoice --input examples\sample_invoice.txt --output output\invoice.json
python main.py --mode invoice
```

## 12. Example input and output

### Illustrative resume result (not a live API response)

Input excerpt from `examples/sample_resume.txt`:

```text
Jordan A. Rivera
jordan.rivera@example.com | +1 (555) 014-7788 | Austin, TX
...
```

Expected validated shape (abridged; full file in
`examples/expected_resume_schema.json`):

```json
{
  "full_name": "Jordan A. Rivera",
  "email": "jordan.rivera@example.com",
  "phone": "+1 (555) 014-7788",
  "professional_summary": "Backend engineer with 6 years of experience...",
  "education": [
    {
      "institution": "University of Texas at Austin",
      "degree": "Bachelor of Science",
      "field_of_study": "Computer Science",
      "start_date": "Aug 2015",
      "end_date": "May 2019",
      "location": "Austin, TX",
      "grade": "3.7/4.0"
    }
  ],
  "technical_skills": ["Python", "Go", "PostgreSQL"],
  "soft_skills": ["Technical mentoring", "cross-team communication"],
  "work_experience": [
    {
      "job_title": "Senior Software Engineer",
      "company": "Northstar Payments",
      "location": "Austin, TX",
      "start_date": "Jan 2022",
      "end_date": null,
      "description": null,
      "highlights": [
        "Led the migration of the settlement service from a monolith to three Go services, reducing p95 latency by 38%."
      ]
    }
  ],
  "projects": [],
  "certifications": ["AWS Certified Solutions Architect - Associate"],
  "languages": [{"language": "English", "proficiency": "Native"}]
}
```

### Illustrative invoice result (not a live API response)

Expected validated shape (abridged; full file in
`examples/expected_invoice_schema.json`):

```json
{
  "invoice_number": "EOS-2025-0042",
  "invoice_date": "2025-03-14",
  "due_date": "2025-04-13",
  "currency": "GBP",
  "supplier_name": "Evergreen Office Supplies Ltd",
  "supplier_address": "Unit 4, 18 Harbour Road, Bristol BS1 4XY, United Kingdom",
  "customer_name": "Harbourview Design Studio",
  "subtotal": 269.9,
  "tax_amount": 53.98,
  "total_amount": 323.88,
  "line_items": [
    {
      "description": "Recycled A4 paper (box of 5 reams)",
      "quantity": 4,
      "unit_price": 24.5,
      "line_total": 98
    }
  ],
  "payment_terms": "Net 30. Pay by bank transfer quoting the invoice number."
}
```

The `$comment` key in each example file explains that these are illustrative
schema references. They are **not** captured API responses. A real result
depends on the provider, model, and document.

### Terminal output shape

On success the CLI prints a short PII-minimizing summary. It prints the complete
JSON only when no `--output` path is supplied; with `--output`, the validated
JSON is written to the file instead:

```text
Extraction succeeded (gemini, model=gemini-3.5-flash, attempts=1).
Summary: invoice: line_items=3, total=323.88, currency=GBP

{
  "invoice_number": "EOS-2025-0042",
  ...
}
Saved validated JSON to: C:\...\output\invoice.json
```

The block above illustrates terminal output without `--output`. When `--output`
is supplied, the JSON body is omitted from the terminal and only the summary and
saved path are printed.

Exit codes: `0` success, `2` usage error, `3` configuration error, `4` input
error, `5` extraction/API error, `6` output write error, `130` cancelled.

## 13. Run the tests

```powershell
python -m pytest -q
python -m pytest -v
python -m pytest tests\test_schemas.py -q
```

Tests are fully offline. Provider SDK clients are replaced with stubs or patched
methods; no API key is read and no request is sent. Expected: `106 passed`.

To see the CLI help:

```powershell
python main.py --help
```

## 14. Common errors and troubleshooting

| Message / symptom | Cause and fix |
| --- | --- |
| `Configuration error: LLM_PROVIDER is not set` | Copy `.env.example` to `.env` and set `LLM_PROVIDER`. |
| `Configuration error: GEMINI_API_KEY is not set` | Set the key for the **selected** provider. |
| `GEMINI_API_KEY still contains the placeholder value` | You copied `.env.example` but did not replace the key. Paste a real key or remove `.env` entirely. |
| `LLM_PROVIDER must be either 'gemini' or 'openai'` | Fix the typo; other providers are not supported. |
| `Gemini authentication or permission failed` | Key is invalid, restricted, or lacks access to the model. |
| `OpenAI authentication failed` | Check `OPENAI_API_KEY` and that it belongs to the intended project. |
| `rate limit or quota exceeded` | Wait and retry, or check provider billing/quota. The app does not loop. |
| `The Gemini request timed out` | Increase `LLM_TIMEOUT_SECONDS` (max 300) or retry later. |
| `OpenAI rejected the request (HTTP 400/404)` | Model name may not support JSON Schema structured outputs. |
| `Extracted data failed schema validation` | The model returned wrong types/values. One retry happens by default, then the run fails. |
| `Input file is not valid UTF-8` | Re-save the document as UTF-8. |
| `The document is too large` | Limit is 200,000 characters / 2,000,000 bytes. |
| `No module named ...` | Activate `.venv` and install `requirements.txt`. |
| PowerShell activation is blocked | Use `.\.venv\Scripts\python.exe ...` directly or the process-scope policy command above. |

## 15. Security and privacy considerations

- **Document content leaves your machine.** The text you provide is sent to the
  selected provider's API. Do not process documents you are not authorized to
  share with that provider.
- Never commit `.env` or real keys. `.gitignore` excludes `.env`, `.env.*`
  (except `.env.example`), key files, virtual environments, caches, and local
  output folders.
- Error messages and summaries avoid document text and credentials. Resume
  summaries report counts only.
- Unknown JSON fields are rejected, and Decimal values are exact rather than
  binary floats.
- The document is wrapped in explicit untrusted-data delimiters and the prompt
  instructs the model to ignore any embedded instructions. This reduces, but
  cannot eliminate, prompt-injection risk; validation is the final gate.
- Temporary output files are written in the destination directory and replaced
  atomically. Generated JSON may still contain personal data, so store and share
  it deliberately.
- Provider retention, logging, and training policies are outside this
  application's control. Review them before using real data.

## 16. Limitations

- Live **Gemini** extraction was executed against `examples/sample_resume.txt`
  and `examples/sample_invoice.txt` using `gemini-3.5-flash`: both produced
  schema-valid JSON. Live **OpenAI** calls were not executed; the OpenAI path is
  verified through mocked SDK tests and offline schema compatibility checks.
- Google retires models for new accounts. `gemini-2.5-flash` and
  `gemini-2.5-pro` now return HTTP 404 ("no longer available to new users").
  The application surfaces Google's explanation, and you can switch models with
  `GEMINI_MODEL`. If a model returns HTTP 503 ("high demand"), retry later or
  choose another current model.
- Results depend on model quality. The app validates structure and types, not
  factual correctness of every extracted value.
- Dates are stored as strings. Partial dates (for example `2021`) are preserved
  in resume fields, while invoice dates require full `YYYY-MM-DD` values or null.
- Currency symbols are not converted to codes when ambiguous; the schema asks
  the model to use null instead of guessing.
- Decimal output is exact JSON text, but any consumer that parses it into a
  binary float can lose precision.
- PDF and DOCX files are not parsed. Convert them to UTF-8 text first.
- The default models are examples and may change availability; override them
  with `GEMINI_MODEL` or `OPENAI_MODEL`.

## 17. GitHub submission checklist

- [ ] `git status` shows only intended project files.
- [ ] `.env` is absent from the repository and from the staging area.
- [ ] `.env.example` contains placeholders only.
- [ ] No API keys, tokens, or personal documents appear in source or history.
- [ ] `python -m pytest -q` passes.
- [ ] `python main.py --help` works.
- [ ] README commands match the submitted project layout.
- [ ] Example files are clearly labelled as illustrative.
- [ ] Your own live run (if any) is documented honestly, including provider and model.
- [ ] No claim of Datheon approval or certification is included.
- [ ] Commit and push only after you have reviewed the diff and are authorized to do so.

