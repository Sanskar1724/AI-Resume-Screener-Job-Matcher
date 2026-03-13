"""Candidate ranking utilities."""

from __future__ import annotations

from typing import Any


def rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort candidates by overall score in descending order and assign rank."""
    sorted_candidates = sorted(
        candidates, key=lambda item: item.get("score", 0.0), reverse=True)
    for index, candidate in enumerate(sorted_candidates, start=1):
        candidate["rank"] = index
    return sorted_candidates
