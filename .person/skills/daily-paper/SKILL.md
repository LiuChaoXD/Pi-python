---
name: daily-paper
description: Use this skill to fetch, review, and save HuggingFace Daily Papers content via CLI tooling.
---

# HuggingFace Daily Papers Skill

This skill helps you stay current with the HuggingFace Daily Papers by fetching summaries, saving metadata, and downloading PDFs.

## Overview

- Fetch the latest papers using the `daily_paper.py` CLI script, which queries `https://huggingface.co/api/daily_papers`, saves metadata to disk, and downloads PDFs when available.
- Store papers, PDF files, and metadata under a configurable directory so you can analyze or share them later.
- Use the generated metadata JSON as input for downstream automation (summaries, ingestion, etc.).

## Scripts

### `scripts/daily_paper.py`

```bash
python scripts/daily_paper.py --limit 10 --save-dir papers
```

- `--limit`: Number of papers (1–100).
- `--save-dir`: Directory to persist metadata (`papers_metadata.json`) and downloaded PDFs.
- Outputs: summary of fetched papers plus a per-file download report.

### `scripts/pdf_parser.py`

```bash
python scripts/pdf_parser.py papers/2602.02474.pdf
```

- Inputs a PDF, calls the Mistral OCR API, and writes a JSON file alongside the PDF.
- Requires `MISTRAL_API_KEY` in the environment (e.g., via a `.env` file).
- Outputs the parsed structure with text, tables, and optional base64 images.

## Workflow Suggestions

1. Run `daily_paper.py` daily to collect the freshest summaries.
2. Inspect `papers/papers_metadata.json` for paper IDs and high-level details.
3. OCR any downloaded PDFs via `pdf_parser.py` when you need searchable text or tables.
4. Feed the JSON outputs into indexing pipelines, note-taking systems, or knowledge graphs.

## Troubleshooting

- If fetching fails, ensure you have network access to `huggingface.co` and retry.
- If OCR fails, verify `MISTRAL_API_KEY` is set and that the PDF is readable (not corrupted).
