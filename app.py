"""CLI entry point for AI Resume Screener & Job Matcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai_insights.explanation import generate_explanation
from data_loader.loader import DatasetLoader
from embeddings.embedding_model import EmbeddingModel
from matching.similarity import compute_cosine_similarity
from matching.skill_gap import compute_skill_gap
from nlp.entity_extractor import EntityExtractor
from nlp.skill_extractor import SkillExtractor
from resume_parser.parser import ResumeParser
from utils.helpers import file_stem


def score_resume_text(
    candidate_name: str,
    resume_text: str,
    job_description: str,
    skill_extractor: SkillExtractor,
    entity_extractor: EntityExtractor,
    embedding_model: EmbeddingModel,
) -> dict[str, Any]:
    """Score one resume text against a job description."""
    resume_skills = skill_extractor.extract_skills(resume_text)
    job_skills = skill_extractor.extract_skills(job_description)

    resume_vector = embedding_model.encode_single(resume_text)
    job_vector = embedding_model.encode_single(job_description)

    similarity_score = compute_cosine_similarity(resume_vector, job_vector)
    gap = compute_skill_gap(resume_skills, job_skills)
    entities = entity_extractor.extract_entities(resume_text)

    explanation = generate_explanation(
        candidate_name=candidate_name,
        similarity_score=similarity_score,
        matched_skills=gap["matched_skills"],
        missing_skills=gap["missing_skills"],
        entities=entities,
    )

    return {
        "candidate": candidate_name,
        "score": similarity_score,
        "matched_skills": gap["matched_skills"],
        "missing_skills": gap["missing_skills"],
        "skill_coverage": gap["skill_coverage"],
        "entities": entities,
        "explanation": explanation,
    }


def run_single_match(resume_pdf: Path, job_description: str) -> dict:
    """Run end-to-end matching for one resume and one job description."""
    parser = ResumeParser()
    loader = DatasetLoader()

    skill_vocab = loader.load_skill_vocabulary()
    skill_extractor = SkillExtractor(skill_vocab)
    entity_extractor = EntityExtractor()
    embedding_model = EmbeddingModel()

    resume_text = parser.extract_text_from_pdf(resume_pdf)
    return score_resume_text(
        candidate_name=file_stem(resume_pdf),
        resume_text=resume_text,
        job_description=job_description,
        skill_extractor=skill_extractor,
        entity_extractor=entity_extractor,
        embedding_model=embedding_model,
    )


def run_dataset_ranking(job_description: str, limit: int = 100) -> list[dict[str, Any]]:
    """Run ranking against resume dataset records for a given job description."""
    loader = DatasetLoader()
    skill_vocab = loader.load_skill_vocabulary()
    skill_extractor = SkillExtractor(skill_vocab)
    entity_extractor = EntityExtractor()
    embedding_model = EmbeddingModel()

    resumes = loader.load_resume_records(nrows=limit)
    results = [
        score_resume_text(
            candidate_name=resume["candidate_name"],
            resume_text=resume["resume_text"],
            job_description=job_description,
            skill_extractor=skill_extractor,
            entity_extractor=entity_extractor,
            embedding_model=embedding_model,
        )
        for resume in resumes
    ]

    return sorted(results, key=lambda item: item.get("score", 0.0), reverse=True)


def main() -> None:
    """CLI wrapper for local testing and automation."""
    parser = argparse.ArgumentParser(
        description="AI Resume Screener & Job Matcher")
    parser.add_argument("--resume", type=Path, required=False,
                        help="Path to candidate resume PDF")
    parser.add_argument("--job", type=str, required=True,
                        help="Job description text")
    parser.add_argument(
        "--dataset-mode",
        action="store_true",
        help="Use resume dataset records instead of --resume PDF",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=100,
        help="Number of resume dataset records to evaluate in dataset mode",
    )
    args = parser.parse_args()

    if args.dataset_mode:
        result = run_dataset_ranking(args.job, limit=args.dataset_limit)
    else:
        if args.resume is None:
            raise ValueError(
                "--resume is required unless --dataset-mode is enabled.")
        result = run_single_match(args.resume, args.job)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
