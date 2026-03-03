#!/usr/bin/env python3
"""CLI for using Mistral OCR to process a PDF and write JSON next to the original file."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

_API_KEY = os.getenv("MISTRAL_API_KEY")
_MODEL = "mistral-ocr-latest"


def _encode_pdf(pdf_path: Path) -> str:
    with pdf_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ocr_pdf(pdf_path: Path, include_images: bool, table_format: str) -> Dict[str, Any]:
    if not _API_KEY:
        raise RuntimeError("MISTRAL_API_KEY is not set")

    client = Mistral(api_key=_API_KEY)
    base64_pdf = _encode_pdf(pdf_path)

    document_payload = {
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{base64_pdf}",
    }

    response = client.ocr.process(
        model=_MODEL,
        document=document_payload,
        include_image_base64=include_images,
        table_format=table_format,
    )

    return json.loads(response.model_dump_json())


def save_output(data: Dict[str, Any], pdf_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fw:
        json.dump(data, fw, ensure_ascii=False, indent=2)


def build_summary(pdf_path: Path, output_path: Path, pages: int) -> None:
    print(f"OCR complete for {pdf_path.name}")
    print(f"Pages processed: {pages}")
    print(f"Result saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a PDF with Mistral OCR and save JSON in place"
    )
    parser.add_argument("--input_pdf", type=str, help="PDF file to OCR")
    parser.add_argument(
        "--table-format",
        choices=["html", "markdown"],
        default="html",
        help="Table output format",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.input_pdf).expanduser()

    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise SystemExit("Input must be an existing PDF file")

    result = ocr_pdf(pdf_path, include_images=False, table_format=args.table_format)
    pages = len(result.get("pages", []))

    output_path = pdf_path.with_suffix(".json")
    save_output(result, pdf_path, output_path)

    build_summary(pdf_path, output_path, pages)


if __name__ == "__main__":
    main()
