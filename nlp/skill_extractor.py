"""Skill extraction logic using a controlled skill vocabulary."""

from __future__ import annotations

from typing import Iterable

from utils.helpers import normalize_text, normalize_token, unique_preserve_order


class SkillExtractor:
    """Extracts skills from text by matching against a skill database."""

    def __init__(self, skill_vocabulary: Iterable[str]) -> None:
        normalized = [normalize_token(skill) for skill in skill_vocabulary]
        self.skill_vocabulary = set(skill for skill in normalized if skill)

    def extract_skills(self, text: str) -> list[str]:
        """Return matched skills found in input text."""
        normalized_text = normalize_token(normalize_text(text))
        if not normalized_text:
            return []

        found: list[str] = []
        tokens = normalized_text.split()

        max_n = 4
        for ngram_size in range(1, max_n + 1):
            for index in range(len(tokens) - ngram_size + 1):
                phrase = " ".join(tokens[index: index + ngram_size])
                if phrase in self.skill_vocabulary:
                    found.append(phrase)

        return unique_preserve_order(found)
