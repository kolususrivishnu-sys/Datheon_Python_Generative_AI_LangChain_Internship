
# Structured LLM Prompt Pipeline & JSON Extractor

A Python Generative AI project that extracts structured information from
invoices and resumes using OpenAI or Google Gemini.

## Features

- Invoice and resume information extraction
- Parameterized prompt templates
- Few-shot examples
- OpenAI and Gemini integrations
- Pydantic schema validation
- JSON output files
- Reusable command-line interface

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` on your computer and add your own API key.
Never publish `.env` or share your secret keys.

## Run the Invoice Extractor

Prepare a plain-text invoice file named `invoice.txt`, then run:

```bash
python app.py --input invoice.txt --type invoice --provider openai \
  --output output.json
```

For Google Gemini:

```bash
python app.py --input invoice.txt --type invoice --provider gemini \
  --output output.json
```

For resume extraction, replace `invoice` with `resume` and provide
the path to a resume text file.

## Validation

Model responses are parsed and validated using Pydantic.
Unexpected fields and invalid data cause validation errors instead
of being silently accepted.

## Security

- Keep API keys in your local `.env` file.
- Do not commit API keys or private resumes and invoices.
- Use fictional sample documents when demonstrating the project.

## Limitations

The input is plain text. Scanned PDFs and images require a separate
OCR or document-text extraction step. Live model calls require a
valid API key and may incur API charges.

## Technologies

Python, OpenAI API, Google Gemini API, Pydantic, JSON.
