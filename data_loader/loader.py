"""Data loading utilities for required datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from config import (
    COMPANIES_PATH,
    COMPANY_INDUSTRIES_PATH,
    JOB_INDUSTRIES_PATH,
    JOB_POSTINGS_PATH,
    JOB_SKILLS_PATH,
    SKILLS_DATASET_PATH,
)
from utils.helpers import normalize_text, normalize_token, unique_preserve_order


class DatasetLoader:
    """Loads and normalizes structured datasets for matching workflows."""

    def __init__(
        self,
        skills_path: Path = SKILLS_DATASET_PATH,
        job_postings_path: Path = JOB_POSTINGS_PATH,
    ) -> None:
        self.skills_path = skills_path
        self.job_postings_path = job_postings_path

    def load_skills_dataframe(self) -> pd.DataFrame:
        """Load skills dataset as a dataframe."""
        return pd.read_csv(self.skills_path)

    def load_resume_dataframe(self, nrows: int | None = None) -> pd.DataFrame:
        """Load resume dataset as a dataframe."""
        from config import RESUME_DATASET_PATH

        return pd.read_csv(RESUME_DATASET_PATH, nrows=nrows)

    def load_job_postings_dataframe(self, nrows: int | None = None) -> pd.DataFrame:
        """Load job postings dataset as a dataframe."""
        return pd.read_csv(self.job_postings_path, nrows=nrows)

    def load_job_skills_dataframe(self, nrows: int | None = None) -> pd.DataFrame:
        """Load job skills bridge dataset."""
        return pd.read_csv(JOB_SKILLS_PATH, nrows=nrows)

    def load_job_industries_dataframe(self, nrows: int | None = None) -> pd.DataFrame:
        """Load job to industry bridge dataset."""
        return pd.read_csv(JOB_INDUSTRIES_PATH, nrows=nrows)

    def load_companies_dataframe(self, nrows: int | None = None) -> pd.DataFrame:
        """Load companies dataset."""
        return pd.read_csv(COMPANIES_PATH, nrows=nrows)

    def load_company_industries_dataframe(self, nrows: int | None = None) -> pd.DataFrame:
        """Load company to industry mapping dataset."""
        return pd.read_csv(COMPANY_INDUSTRIES_PATH, nrows=nrows)

    def load_skill_vocabulary(self) -> list[str]:
        """Build a normalized skill vocabulary from the skills dataset."""
        df = self.load_skills_dataframe()

        candidate_columns = [
            "skill name",
            "skill",
            "skills",
            "technical_skill",
            "example",
            "name",
            "technology",
        ]

        selected_column = None
        lower_to_original = {col.lower(): col for col in df.columns}
        for column in candidate_columns:
            if column in lower_to_original:
                selected_column = lower_to_original[column]
                break

        if selected_column is None:
            object_columns = [
                col for col in df.columns if df[col].dtype == "object"]
            if not object_columns:
                raise ValueError("No textual column found in skills dataset.")
            selected_column = object_columns[0]

        skills = (
            df[selected_column]
            .dropna()
            .astype(str)
            .map(str.strip)
            .loc[lambda series: series != ""]
            .tolist()
        )

        normalized = [normalize_token(skill) for skill in skills]
        normalized = [skill for skill in normalized if skill]
        return unique_preserve_order(normalized)

    @staticmethod
    def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str:
        lower_to_original = {col.lower(): col for col in df.columns}
        for candidate in candidates:
            if candidate.lower() in lower_to_original:
                return lower_to_original[candidate.lower()]
        raise ValueError(f"Could not resolve any of columns: {candidates}")

    @staticmethod
    def _resolve_column_optional(df: pd.DataFrame, candidates: list[str]) -> str | None:
        try:
            return DatasetLoader._resolve_column(df, candidates)
        except ValueError:
            return None

    def load_job_records(self, nrows: int = 500, min_description_chars: int = 80) -> list[dict]:
        """Load job records with robust column resolution across job datasets."""
        df = self.load_job_postings_dataframe(nrows=nrows)
        job_skills_df = self.load_job_skills_dataframe()
        job_industries_df = self.load_job_industries_dataframe()
        companies_df = self.load_companies_dataframe()
        company_industries_df = self.load_company_industries_dataframe()

        title_col = self._resolve_column(df, ["title", "job_title", "name"])
        description_col = self._resolve_column(
            df, ["description", "job_description", "skills_desc"])
        id_col = self._resolve_column(df, ["job_id", "id"])
        company_id_col = self._resolve_column_optional(
            df, ["company_id", "company"])
        location_col = self._resolve_column_optional(df, ["location", "city"])
        experience_col = self._resolve_column_optional(
            df, ["formatted_experience_level", "experience_level", "seniority"])
        work_type_col = self._resolve_column_optional(
            df, ["formatted_work_type", "work_type"])

        job_skill_id_col = self._resolve_column(
            job_skills_df, ["job_id", "id"])
        job_skill_value_col = self._resolve_column(
            job_skills_df, ["skill_abr", "skill", "skill_name", "skills"])

        job_industry_job_id_col = self._resolve_column(
            job_industries_df, ["job_id", "id"])
        job_industry_value_col = self._resolve_column(
            job_industries_df, ["industry", "industry_id"])

        companies_id_col = self._resolve_column(
            companies_df, ["company_id", "id"])
        companies_name_col = self._resolve_column(
            companies_df, ["name", "company_name"])

        company_industries_company_col = self._resolve_column(
            company_industries_df, ["company_id", "id"])
        company_industries_value_col = self._resolve_column(
            company_industries_df, ["industry", "industry_name"])

        skills_map: dict[str, list[str]] = {}
        for _, row in job_skills_df.iterrows():
            key = str(row.get(job_skill_id_col, "") or "").strip()
            value = str(row.get(job_skill_value_col, "") or "").strip()
            if not key or not value:
                continue
            skills_map.setdefault(key, []).append(value)

        job_industry_map: dict[str, list[str]] = {}
        for _, row in job_industries_df.iterrows():
            key = str(row.get(job_industry_job_id_col, "") or "").strip()
            value = str(row.get(job_industry_value_col, "") or "").strip()
            if not key or not value:
                continue
            job_industry_map.setdefault(key, []).append(value)

        company_name_map: dict[str, str] = {}
        for _, row in companies_df.iterrows():
            key = str(row.get(companies_id_col, "") or "").strip()
            value = str(row.get(companies_name_col, "") or "").strip()
            if key and value:
                company_name_map[key] = value

        company_industry_map: dict[str, list[str]] = {}
        for _, row in company_industries_df.iterrows():
            key = str(row.get(company_industries_company_col, "") or "").strip()
            value = str(row.get(company_industries_value_col, "")
                        or "").strip()
            if not key or not value:
                continue
            company_industry_map.setdefault(key, []).append(value)

        records = []
        seen_ids: set[str] = set()
        for _, row in df.iterrows():
            raw_description = str(row.get(description_col, "") or "").strip()
            description = normalize_text(raw_description)
            if len(description) < min_description_chars:
                continue

            job_id = str(row.get(id_col, "") or "").strip()
            if not job_id or job_id in seen_ids:
                continue
            seen_ids.add(job_id)

            company_id = str(row.get(company_id_col, "")
                             or "").strip() if company_id_col else ""
            company_name = company_name_map.get(company_id, "")
            job_skills = unique_preserve_order(skills_map.get(job_id, []))
            job_industries = unique_preserve_order(
                job_industry_map.get(job_id, []))
            company_industries = unique_preserve_order(
                company_industry_map.get(company_id, []))

            context_parts: list[str] = []
            if job_skills:
                context_parts.append(
                    f"Required skills: {', '.join(job_skills)}")
            if job_industries:
                context_parts.append(
                    f"Job industries: {', '.join(job_industries)}")
            if company_industries:
                context_parts.append(
                    f"Company industries: {', '.join(company_industries)}")

            matching_text = description
            if context_parts:
                matching_text = f"{description}\n\n" + "\n".join(context_parts)

            records.append(
                {
                    "job_id": job_id,
                    "title": str(row.get(title_col, "") or "Untitled Role").strip(),
                    "description": description,
                    "company_id": company_id,
                    "company_name": company_name,
                    "location": str(row.get(location_col, "") or "").strip() if location_col else "",
                    "experience_level": str(row.get(experience_col, "") or "").strip() if experience_col else "",
                    "work_type": str(row.get(work_type_col, "") or "").strip() if work_type_col else "",
                    "skills": job_skills,
                    "job_industries": job_industries,
                    "company_industries": company_industries,
                    "matching_text": matching_text,
                }
            )
        return records

    def load_resume_records(self, nrows: int = 300) -> list[dict]:
        """Load candidate resume records from resume dataset."""
        df = self.load_resume_dataframe(nrows=nrows)

        id_col = self._resolve_column(df, ["id", "resume_id"])
        text_col = self._resolve_column(
            df, ["resume_str", "resume_text", "text"])
        category_col = None
        try:
            category_col = self._resolve_column(
                df, ["category", "label", "domain"])
        except ValueError:
            category_col = None

        records = []
        for _, row in df.iterrows():
            text = str(row.get(text_col, "") or "").strip()
            if not text:
                continue

            payload = {
                "candidate_id": str(row.get(id_col, "")),
                "candidate_name": f"Candidate {row.get(id_col, '')}",
                "resume_text": text,
            }
            if category_col is not None:
                payload["category"] = str(
                    row.get(category_col, "") or "").strip()
            records.append(payload)

        return records

    @staticmethod
    def serialize_records(df: pd.DataFrame, fields: Iterable[str]) -> list[dict]:
        """Convert selected columns to list-of-dicts payload."""
        missing = [field for field in fields if field not in df.columns]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        return df[list(fields)].to_dict(orient="records")

    def dataset_quality_summary(self, resume_nrows: int = 500, job_nrows: int = 1000) -> dict:
        """Compute lightweight quality metrics over key datasets."""
        resumes = self.load_resume_dataframe(nrows=resume_nrows)
        jobs = self.load_job_postings_dataframe(nrows=job_nrows)
        jobs_clean = self.load_job_records(nrows=job_nrows)

        resume_text_col = self._resolve_column(
            resumes, ["resume_str", "resume_text", "text"])
        job_description_col = self._resolve_column(
            jobs, ["description", "job_description", "skills_desc"])
        job_id_col = self._resolve_column(jobs, ["job_id", "id"])

        raw_resume_count = len(resumes)
        raw_job_count = len(jobs)
        cleaned_job_count = len(jobs_clean)
        resume_non_empty = int((resumes[resume_text_col].fillna(
            "").astype(str).str.strip() != "").sum())
        job_non_empty = int((jobs[job_description_col].fillna(
            "").astype(str).str.strip() != "").sum())
        unique_job_ids = int(jobs[job_id_col].astype(str).nunique())

        return {
            "raw_resume_records": raw_resume_count,
            "resumes_with_text": resume_non_empty,
            "raw_job_records": raw_job_count,
            "jobs_with_description": job_non_empty,
            "unique_job_ids": unique_job_ids,
            "cleaned_job_records": cleaned_job_count,
        }
