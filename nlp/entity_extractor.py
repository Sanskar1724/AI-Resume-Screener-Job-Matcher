"""Entity extraction for resume metadata (education, experience, organizations)."""

from __future__ import annotations

import re
from typing import Any

from config import SPACY_MODEL_NAME
from utils.helpers import normalize_text, unique_preserve_order


EXPERIENCE_PATTERN = re.compile(
    r"(\d+\+?\s*(?:years?|yrs?))\s+of\s+experience", re.IGNORECASE
)


class EntityExtractor:
    """Extract high-value entities from resume text using spaCy + heuristics."""

    def __init__(self, model_name: str = SPACY_MODEL_NAME) -> None:
        self.nlp = self._load_model(model_name)

    @staticmethod
    def _load_model(model_name: str) -> Any:
        try:
            import spacy

            return spacy.load(model_name)
        except Exception:
            return None

    def extract_entities(self, text: str) -> dict:
        """Extract education, organizations, locations, and experience mentions."""
        cleaned = normalize_text(text)

        education_keywords = {
            "bachelor",
            "master",
            "phd",
            "b.sc",
            "m.sc",
            "mba",
            "bs",
            "ms",
            "university",
            "college",
        }

        if self.nlp is not None:
            doc = self.nlp(cleaned)
            sentences = [sentence.text.strip() for sentence in doc.sents] if doc.has_annotation(
                "SENT_START") else cleaned.split(".")
            orgs = [entity.text for entity in doc.ents if entity.label_ == "ORG"]
            locations = [
                entity.text for entity in doc.ents if entity.label_ in {"GPE", "LOC"}]
        else:
            sentences = cleaned.split(".")
            orgs = []
            locations = []

        education_lines = [
            sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in education_keywords)
        ]

        experience = EXPERIENCE_PATTERN.findall(cleaned)

        return {
            "education": unique_preserve_order([line for line in education_lines if line]),
            "organizations": unique_preserve_order(orgs),
            "locations": unique_preserve_order(locations),
            "experience_mentions": unique_preserve_order(experience),
        }
