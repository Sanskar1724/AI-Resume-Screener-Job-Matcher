"""Skill gap analysis utilities."""

from __future__ import annotations

from utils.helpers import unique_preserve_order


def compute_skill_gap(resume_skills: list[str], job_skills: list[str]) -> dict:
    """Compute overlapping and missing skills between resume and job description."""
    resume_set = set(resume_skills)
    job_set = set(job_skills)

    matched = unique_preserve_order(
        [skill for skill in job_skills if skill in resume_set])
    missing = unique_preserve_order(
        [skill for skill in job_skills if skill not in resume_set])

    coverage = 0.0 if not job_set else round(
        (len(matched) / len(job_set)) * 100, 2)

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "skill_coverage": coverage,
    }
