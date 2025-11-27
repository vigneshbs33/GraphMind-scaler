"""Utilities for parsing and chunking uploaded files."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

import xmltodict
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


class MultiFormatParser:
    """Static helpers for reading multiple document formats."""

    @staticmethod
    def parse_file(file_path: Path, file_type: str) -> Dict[str, str]:
        """Return a dict containing extracted content or an error."""
        try:
            if file_type == "text":
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            elif file_type == "pdf":
                reader = PdfReader(str(file_path))
                content = "\n".join(page.extract_text() or "" for page in reader.pages)
            elif file_type == "xml":
                data = xmltodict.parse(file_path.read_text(encoding="utf-8", errors="ignore"))
                content = json.dumps(data, indent=2)
            elif file_type == "json":
                data = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
                content = json.dumps(data, indent=2)
            elif file_type == "csv":
                rows = []
                with file_path.open(newline="", encoding="utf-8", errors="ignore") as handle:
                    reader = csv.reader(handle)
                    for row in reader:
                        rows.append(", ".join(row))
                content = "\n".join(rows)
            else:
                return {"content": "", "error": f"Unsupported file type: {file_type}"}
            return {"content": content.strip(), "error": ""}
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to parse %s: %s", file_path, exc)
            return {"content": "", "error": str(exc)}

    @staticmethod
    def extract_entities(content: str) -> List[str]:
        """Return unique capitalized tokens that resemble entities."""
        candidates = set()
        for token in content.replace("\n", " ").split():
            token = token.strip(".,;:()[]{}\"'")
            if len(token) >= 3 and token[0].isupper():
                candidates.add(token)
        return sorted(list(candidates))[:20]

    @staticmethod
    def chunk_text(content: str, chunk_size: int = 500) -> List[str]:
        """Split text into overlapping chunks respecting sentence boundaries."""
        sentences = [s.strip() for s in content.split(".") if s.strip()]
        if not sentences:
            return [content[:chunk_size]]

        chunks: List[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}.".strip()
            if len(candidate) >= chunk_size:
                if current:
                    chunks.append(current.strip())
                current = sentence + "."
            else:
                current = candidate
        if current:
            chunks.append(current.strip())

        # Add overlap
        overlapped: List[str] = []
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                overlapped.append(chunk)
                continue
            prev_tail = overlapped[-1][-50:]
            overlapped.append(f"{prev_tail} {chunk}".strip())
        return overlapped or [content]

