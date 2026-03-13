"""Utility helpers used across modules."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9+#.\- ]+")


def normalize_text(text: str) -> str:
    """Normalize free-form text for reliable downstream processing."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def normalize_token(text: str) -> str:
    """Normalize token-like values for exact matching."""
    cleaned = NON_ALNUM_PATTERN.sub(" ", text.lower())
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip()


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    """Deduplicate strings while preserving original insertion order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def file_stem(path_like: str | Path) -> str:
    """Return a clean display name derived from a path."""
    return Path(path_like).stem
