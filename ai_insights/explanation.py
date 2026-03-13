"""Human-readable AI explanation generation."""

from __future__ import annotations


def generate_explanation(
    candidate_name: str,
    similarity_score: float,
    matched_skills: list[str],
    missing_skills: list[str],
    entities: dict,
) -> str:
    """Generate concise recruiter-facing explanation text."""
    strengths = ", ".join(
        matched_skills[:8]) if matched_skills else "No direct skill overlap detected"
    gaps = ", ".join(
        missing_skills[:8]) if missing_skills else "No critical skill gaps detected"

    education_summary = "; ".join(entities.get("education", [])[:2])
    if not education_summary:
        education_summary = "Education details were limited in extracted text"

    experience_summary = ", ".join(entities.get("experience_mentions", [])[:3])
    if not experience_summary:
        experience_summary = "No explicit years-of-experience mention was detected"

    return (
        f"{candidate_name} has a {similarity_score:.2f}% profile match with the job description. "
        f"Strongly aligned skills: {strengths}. "
        f"Primary skill gaps: {gaps}. "
        f"Education signal: {education_summary}. "
        f"Experience signal: {experience_summary}."
    )
