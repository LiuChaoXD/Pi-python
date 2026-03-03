#!/usr/bin/env python3
"""Simple script to fetch HuggingFace daily papers and save their PDFs."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# check env contain DEFAULT_MODEL
if "WORKSPACE" not in os.environ:
    raise EnvironmentError("WORKSPACE not set in environment variables.")

WORKSPACE = os.environ["WORKSPACE"]
import requests

# HuggingFace Daily Papers API
API_URL = "https://huggingface.co/api/daily_papers"


@dataclass
class Paper:
    id: str
    title: str
    authors: List[str]
    summary: str
    upvotes: int
    num_comments: int
    published_at: str
    thumbnail: str
    huggingface_url: str
    arxiv_url: str
    pdf_url: str


def _extract_paper_info(raw: Dict[str, Any]) -> Paper:
    paper_data = raw.get("paper", {})
    paper_id = paper_data.get("id", "")

    authors = [
        a.get("name", "") for a in paper_data.get("authors", []) if a.get("name")
    ]

    return Paper(
        id=paper_id,
        title=raw.get("title", "No title"),
        authors=authors,
        summary=raw.get("summary", ""),
        upvotes=int(paper_data.get("upvotes", 0) or 0),
        num_comments=int(raw.get("numComments", 0) or 0),
        published_at=raw.get("publishedAt", ""),
        thumbnail=raw.get("thumbnail", ""),
        huggingface_url=f"https://huggingface.co/papers/{paper_id}",
        arxiv_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}.pdf",
    )


def fetch_daily_papers(limit: int = 10) -> List[Paper]:
    limit = min(max(limit, 1), 100)
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "daily-paper-fetcher/1.0", "Accept": "application/json"}
    )

    resp = session.get(f"{API_URL}?limit={limit}", timeout=30)
    resp.raise_for_status()
    data = resp.json()

    return [_extract_paper_info(item) for item in data or []]


def download_arxiv_pdf(
    paper_id: str, save_dir: Path, filename: Optional[str] = None
) -> Dict[str, Any]:
    paper_id = paper_id.strip()
    filename = filename or f"{paper_id}.pdf"
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / filename

    if file_path.exists():
        return {"status": "exists", "filename": filename, "path": str(file_path)}

    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    session = requests.Session()
    session.headers.update({"User-Agent": "daily-paper-fetcher/1.0"})

    with session.get(pdf_url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(file_path, "wb") as fw:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fw.write(chunk)

    return {"status": "downloaded", "filename": filename, "path": str(file_path)}


def save_metadata(papers: List[Paper], save_dir: Path) -> Path:
    metadata_path = save_dir / "papers_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fw:
        json.dump(
            [paper.__dict__ for paper in papers], fw, ensure_ascii=False, indent=2
        )
    return metadata_path


def build_summary(
    papers: List[Paper],
    download_results: List[Dict[str, Any]],
    metadata_path: Optional[Path],
) -> None:
    print(f"Fetched {len(papers)} daily papers")
    if metadata_path:
        print(f"Metadata saved to: {metadata_path}")
    if not download_results:
        print("No PDFs downloaded")
        return

    print("Downloaded PDFs:")
    for entry in download_results:
        print(f"- {entry['filename']}: {entry['path']} ({entry['status']})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch HuggingFace daily papers and download PDFs"
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of daily papers to fetch (1-100)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="papers", help="Directory for metadata and PDFs"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Resolve workspace root
    workspace = Path(WORKSPACE).resolve()

    # Resolve directory path
    dir_path_obj = Path(args.save_dir)
    if not dir_path_obj.is_absolute():
        resolved_dir = (workspace / dir_path_obj).resolve()
    else:
        resolved_dir = dir_path_obj.resolve()

    if not resolved_dir.exists():
        resolved_dir.mkdir(parents=True, exist_ok=True)

    try:
        papers = fetch_daily_papers(limit=args.limit)
    except requests.RequestException as exc:
        raise SystemExit(f"Failed to fetch daily papers: {exc}")

    metadata_path = save_metadata(papers, resolved_dir)
    download_results = []

    for paper in papers:
        if not paper.id:
            continue
        try:
            result = download_arxiv_pdf(paper.id, resolved_dir)
        except requests.RequestException as exc:
            result = {
                "status": "failed",
                "filename": f"{paper.id}.pdf",
                "path": str(resolved_dir / f"{paper.id}.pdf"),
                "error": str(exc),
            }
        download_results.append(result)

    build_summary(papers, download_results, metadata_path)


if __name__ == "__main__":
    main()
