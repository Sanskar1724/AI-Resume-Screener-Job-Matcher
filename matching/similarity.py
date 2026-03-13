"""Similarity computation for resume-to-job matching."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import SIMILARITY_SCALE


def compute_cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Compute cosine similarity between two dense vectors as a percentage."""
    arr_a = np.array(vector_a, dtype=float).reshape(1, -1)
    arr_b = np.array(vector_b, dtype=float).reshape(1, -1)
    score = float(cosine_similarity(arr_a, arr_b)[0][0])
    score = max(0.0, min(1.0, score))
    return round(score * SIMILARITY_SCALE, 2)
