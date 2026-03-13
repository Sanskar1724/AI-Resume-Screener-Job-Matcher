"""Global configuration for the AI Resume Screener & Job Matcher."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = ROOT_DIR.parent / "datasets"

SKILLS_DATASET_PATH = DATASETS_DIR / "Skills Dataset" / "technical_skills.csv"
JOB_POSTINGS_PATH = DATASETS_DIR / "LinkedIn Job Dataset" / "job_postings.csv"
JOB_SKILLS_PATH = DATASETS_DIR / "LinkedIn Job Dataset" / "job_skills.csv"
JOB_INDUSTRIES_PATH = DATASETS_DIR / "LinkedIn Job Dataset" / "job_industries.csv"
COMPANIES_PATH = DATASETS_DIR / "LinkedIn Job Dataset" / "companies.csv"
COMPANY_INDUSTRIES_PATH = DATASETS_DIR / \
    "LinkedIn Job Dataset" / "company_industries.csv"
RESUME_DATASET_PATH = DATASETS_DIR / "Resume" / "Resume.csv"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPACY_MODEL_NAME = "en_core_web_sm"

SIMILARITY_SCALE = 100.0
DEFAULT_TOP_K = 10

ALLOWED_RESUME_EXTENSIONS = {".pdf"}

DASHBOARD_TITLE = "AI Resume Screener & Job Matcher"
