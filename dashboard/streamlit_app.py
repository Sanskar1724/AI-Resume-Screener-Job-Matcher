"""Streamlit recruiter dashboard for resume screening and candidate ranking."""

from __future__ import annotations
from utils.helpers import file_stem
from resume_parser.parser import ResumeParser
from nlp.skill_extractor import SkillExtractor
from nlp.entity_extractor import EntityExtractor
from matching.skill_gap import compute_skill_gap
from matching.ranking import rank_candidates
from embeddings.embedding_model import EmbeddingModel
from data_loader.loader import DatasetLoader
from config import DASHBOARD_TITLE
from ai_insights.explanation import generate_explanation

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


@st.cache_resource
def get_runtime_components() -> dict[str, Any]:
    """Initialize and cache heavy components once per app session."""
    loader = DatasetLoader()
    skill_vocab = loader.load_skill_vocabulary()
    return {
        "loader": loader,
        "resume_parser": ResumeParser(),
        "skill_extractor": SkillExtractor(skill_vocab),
        "entity_extractor": EntityExtractor(),
        "embedding_model": EmbeddingModel(),
    }


def apply_page_style() -> None:
    """Apply lightweight visual refinements for a cleaner recruiter UI."""
    st.markdown(
        """
        <style>
            .block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
            .metric-card {
                border: 1px solid rgba(49, 51, 63, 0.2);
                border-radius: 14px;
                padding: 0.9rem 1rem;
                background: rgba(248, 249, 251, 0.35);
            }
            .candidate-card {
                border: 1px solid rgba(49, 51, 63, 0.16);
                border-radius: 14px;
                padding: 0.75rem 0.9rem;
                margin-bottom: 0.6rem;
                background: rgba(248, 249, 251, 0.28);
            }
            .chip-wrap {
                display: flex;
                flex-wrap: wrap;
                gap: 0.35rem;
                margin-top: 0.3rem;
            }
            .chip-match {
                display: inline-block;
                border-radius: 999px;
                padding: 0.18rem 0.62rem;
                font-size: 0.78rem;
                border: 1px solid rgba(34, 197, 94, 0.45);
                background: rgba(34, 197, 94, 0.12);
            }
            .chip-gap {
                display: inline-block;
                border-radius: 999px;
                padding: 0.18rem 0.62rem;
                font-size: 0.78rem;
                border: 1px solid rgba(245, 158, 11, 0.45);
                background: rgba(245, 158, 11, 0.12);
            }
            .muted-text {color: #6b7280; font-size: 0.85rem;}
            .section-caption {color: #6b7280; font-size: 0.9rem; margin-top: -0.25rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_skill_chips(skills: list[str], is_gap: bool = False, limit: int = 18) -> None:
    """Render compact visual chips for skill lists."""
    if not skills:
        st.info("No skills found.")
        return

    css_class = "chip-gap" if is_gap else "chip-match"
    chips = "".join(
        [f"<span class='{css_class}'>{skill}</span>" for skill in skills[:limit]]
    )
    st.markdown(
        f"<div class='chip-wrap'>{chips}</div>", unsafe_allow_html=True)


def render_top_candidate_showcase(ranked: list[dict[str, Any]]) -> None:
    """Render highlight cards for top candidates for fast recruiter scan."""
    if not ranked:
        return

    st.markdown("### Top Candidates Snapshot")
    top_items = ranked[:3]
    cols = st.columns(len(top_items))

    for col, item in zip(cols, top_items):
        with col:
            st.markdown(
                (
                    "<div class='candidate-card'>"
                    f"<strong>#{item['rank']} {item['candidate']}</strong><br/>"
                    f"<span class='muted-text'>Match Score</span><br/><strong>{item['score']:.2f}%</strong><br/>"
                    f"<span class='muted-text'>Skill Coverage</span><br/><strong>{item['skill_coverage']:.2f}%</strong>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def process_candidate_text(
    candidate_name: str,
    resume_text: str,
    job_description: str,
    components: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Process one resume text entry against a job description."""
    skill_extractor = components["skill_extractor"]
    entity_extractor = components["entity_extractor"]
    embedding_model = components["embedding_model"]

    resume_skills = skill_extractor.extract_skills(resume_text)
    job_skills = skill_extractor.extract_skills(job_description)

    resume_vector = np.array(
        embedding_model.encode_single(resume_text), dtype=float)
    job_vector = np.array(embedding_model.encode_single(
        job_description), dtype=float)

    raw_score = float(np.dot(resume_vector, job_vector))
    score = round(max(0.0, min(1.0, raw_score)) * 100.0, 2)

    gap = compute_skill_gap(resume_skills, job_skills)
    entities = entity_extractor.extract_entities(resume_text)

    explanation = generate_explanation(
        candidate_name=candidate_name,
        similarity_score=score,
        matched_skills=gap["matched_skills"],
        missing_skills=gap["missing_skills"],
        entities=entities,
    )

    result = {
        "candidate": candidate_name,
        "score": score,
        "skill_coverage": gap["skill_coverage"],
        "matched_skills": gap["matched_skills"],
        "missing_skills": gap["missing_skills"],
        "entities": entities,
        "explanation": explanation,
    }
    if extra:
        result.update(extra)
    return result


def process_uploaded_resume(uploaded_file, job_description: str, components: dict[str, Any]) -> dict[str, Any]:
    """Process one uploaded resume against job description."""
    resume_parser = components["resume_parser"]
    candidate_name = file_stem(uploaded_file.name)
    resume_text = resume_parser.extract_text_from_uploaded_bytes(
        uploaded_file.getvalue())
    return process_candidate_text(candidate_name, resume_text, job_description, components)


def build_missing_skill_frame(ranked: list[dict[str, Any]], top_n: int = 12) -> pd.DataFrame:
    """Build a frequency table for missing skills across ranked candidates."""
    counter: Counter[str] = Counter()
    for item in ranked:
        counter.update(item.get("missing_skills", []))

    rows = [{"Skill": skill, "Count": count}
            for skill, count in counter.most_common(top_n)]
    return pd.DataFrame(rows)


def build_matched_skill_frame(ranked: list[dict[str, Any]], top_n: int = 12) -> pd.DataFrame:
    """Build a frequency table for matched skills across ranked candidates."""
    counter: Counter[str] = Counter()
    for item in ranked:
        counter.update(item.get("matched_skills", []))

    rows = [{"Skill": skill, "Count": count}
            for skill, count in counter.most_common(top_n)]
    return pd.DataFrame(rows)


def render_overview_metrics(ranked: list[dict[str, Any]]) -> None:
    """Render primary KPI metrics for candidate screening output."""
    if not ranked:
        return

    score_values = [item["score"] for item in ranked]
    coverage_values = [item["skill_coverage"] for item in ranked]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Candidates", len(ranked))
    col2.metric("Top Match", f"{max(score_values):.2f}%")
    col3.metric("Avg Match", f"{np.mean(score_values):.2f}%")
    col4.metric("Avg Skill Coverage", f"{np.mean(coverage_values):.2f}%")


def rank_roles_for_resume(
    resume_text: str,
    candidate_name: str,
    job_records: list[dict[str, Any]],
    components: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    """Match one resume against multiple roles and return top ranked job matches."""
    if not job_records:
        return []

    embedding_model = components["embedding_model"]
    skill_extractor = components["skill_extractor"]
    entity_extractor = components["entity_extractor"]

    resume_vector = np.array(
        embedding_model.encode_single(resume_text), dtype=float)
    job_texts = [item.get("matching_text", item.get(
        "description", "")) for item in job_records]
    job_vectors = np.array(embedding_model.encode(job_texts), dtype=float)

    raw_scores = np.matmul(job_vectors, resume_vector)
    clipped_scores = np.clip(raw_scores, 0.0, 1.0) * 100.0

    top_indices = np.argsort(clipped_scores)[::-1][:top_k]

    resume_skills = skill_extractor.extract_skills(resume_text)
    results: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_indices, start=1):
        job = job_records[int(idx)]
        score = float(round(clipped_scores[int(idx)], 2))

        job_skills = skill_extractor.extract_skills(
            job.get("matching_text", ""))
        gap = compute_skill_gap(resume_skills, job_skills)
        entities = entity_extractor.extract_entities(resume_text)

        explanation = generate_explanation(
            candidate_name=candidate_name,
            similarity_score=score,
            matched_skills=gap["matched_skills"],
            missing_skills=gap["missing_skills"],
            entities=entities,
        )

        results.append(
            {
                "rank": rank,
                "job_id": job.get("job_id", ""),
                "title": job.get("title", ""),
                "company": job.get("company_name", ""),
                "location": job.get("location", ""),
                "work_type": job.get("work_type", ""),
                "experience_level": job.get("experience_level", ""),
                "score": score,
                "skill_coverage": gap["skill_coverage"],
                "matched_skills": gap["matched_skills"],
                "missing_skills": gap["missing_skills"],
                "job_industries": job.get("job_industries", []),
                "company_industries": job.get("company_industries", []),
                "explanation": explanation,
            }
        )

    return results


def render_candidate_screening_tab(components: dict[str, Any]) -> None:
    """Render candidate ranking workflow (many candidates vs one role)."""
    loader = components["loader"]

    st.subheader("Candidate Screening")
    st.caption(
        "Compare many resumes against one target role and generate recruiter-ready analysis.")

    left, right = st.columns([1, 1])

    with left:
        candidate_source = st.radio(
            "Candidate source",
            options=["Upload Resume PDFs", "Use Resume Dataset"],
            horizontal=True,
        )

        uploaded_resumes = []
        dataset_resumes_limit = 100
        if candidate_source == "Upload Resume PDFs":
            uploaded_resumes = st.file_uploader(
                "Upload one or more resume PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                key="screening_uploader",
            )
        else:
            dataset_resumes_limit = st.slider(
                "Resume records", min_value=20, max_value=1000, value=150, step=10, key="screening_resume_limit"
            )

    with right:
        job_source = st.radio(
            "Job description source",
            options=["Paste Manually", "Use Job Postings Dataset"],
            horizontal=True,
        )

        job_description = ""
        if job_source == "Paste Manually":
            job_description = st.text_area(
                "Target job description",
                height=210,
                placeholder="Paste job description here...",
                key="screening_job_text",
            )
            selected_job_title = "Custom Job Description"
        else:
            jobs_limit = st.slider("Job records", min_value=100, max_value=3000,
                                   value=500, step=100, key="screening_job_limit")
            job_records = loader.load_job_records(nrows=jobs_limit)
            if not job_records:
                st.error("No valid jobs with descriptions were found in dataset.")
                return

            labels = [
                f"{item['job_id']} | {item['title']} | {item.get('company_name', '')}".strip(
                )
                for item in job_records
            ]
            selected_label = st.selectbox(
                "Select target role", options=labels, key="screening_job_select")
            selected_job = job_records[labels.index(selected_label)]
            selected_job_title = selected_job["title"]
            job_description = selected_job.get(
                "matching_text", selected_job["description"])

            context = [selected_job.get("location", ""), selected_job.get(
                "work_type", ""), selected_job.get("experience_level", "")]
            context = [item for item in context if item]
            if context:
                st.caption(" | ".join(context))
            if selected_job.get("job_industries"):
                st.caption(
                    f"Job industries: {', '.join(selected_job['job_industries'][:8])}")

    run_button = st.button("Run Candidate Screening", type="primary")
    if not run_button:
        return

    if candidate_source == "Upload Resume PDFs" and not uploaded_resumes:
        st.warning("Please upload at least one resume PDF.")
        return
    if not job_description.strip():
        st.warning("Please provide a job description.")
        return

    with st.spinner("Running screening and generating analysis..."):
        if candidate_source == "Upload Resume PDFs":
            candidate_results = [
                process_uploaded_resume(
                    uploaded_file, job_description, components)
                for uploaded_file in uploaded_resumes
            ]
        else:
            resume_records = loader.load_resume_records(
                nrows=dataset_resumes_limit)
            candidate_results = [
                process_candidate_text(
                    candidate_name=record["candidate_name"],
                    resume_text=record["resume_text"],
                    job_description=job_description,
                    components=components,
                    extra={"category": record.get("category", "")},
                )
                for record in resume_records
            ]

        ranked = rank_candidates(candidate_results)

    st.markdown(f"### Results for: {selected_job_title}")
    render_overview_metrics(ranked)
    render_top_candidate_showcase(ranked)

    ranking_df = pd.DataFrame(
        [
            {
                "Rank": item["rank"],
                "Candidate": item["candidate"],
                "Match Score (%)": item["score"],
                "Skill Coverage (%)": item["skill_coverage"],
                "Category": item.get("category", ""),
            }
            for item in ranked
        ]
    )
    st.dataframe(ranking_df, width="stretch", hide_index=True)

    score_plot_df = ranking_df.head(15).copy()
    if not score_plot_df.empty:
        st.markdown("### Score Analysis")
        st.bar_chart(score_plot_df.set_index("Candidate")[
                     ["Match Score (%)", "Skill Coverage (%)"]])

    missing_df = build_missing_skill_frame(ranked)
    matched_df = build_matched_skill_frame(ranked)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Top Skill Gaps")
        if missing_df.empty:
            st.info("No missing skills identified.")
        else:
            st.bar_chart(missing_df.set_index("Skill"))
    with col2:
        st.markdown("### Top Matched Skills")
        if matched_df.empty:
            st.info("No matched skills identified.")
        else:
            st.bar_chart(matched_df.set_index("Skill"))

    st.markdown("### Candidate Profiles")
    for item in ranked[:20]:
        with st.expander(f"#{item['rank']} - {item['candidate']} ({item['score']}%)", expanded=item["rank"] == 1):
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.metric("Match Score", f"{item['score']:.2f}%")
            with c2:
                st.metric("Skill Coverage", f"{item['skill_coverage']:.2f}%")
            with c3:
                st.progress(
                    min(max(item["score"] / 100.0, 0.0), 1.0), text="Overall Match Strength")

            tab_summary, tab_skills, tab_explain = st.tabs(
                ["Summary", "Skills", "AI Insight"])
            with tab_summary:
                if item.get("category"):
                    st.caption(f"Resume category: {item['category']}")
                entities = item.get("entities", {})
                edu = entities.get("education", [])[:2]
                exp = entities.get("experience_mentions", [])[:3]
                org = entities.get("organizations", [])[:3]

                st.markdown("**Education Signals**")
                st.write(edu or ["No education signals extracted"])
                st.markdown("**Experience Signals**")
                st.write(exp or ["No explicit experience duration found"])
                st.markdown("**Organization Signals**")
                st.write(org or ["No organization entities extracted"])

            with tab_skills:
                st.markdown("**Matched Skills**")
                render_skill_chips(item["matched_skills"], is_gap=False)
                st.markdown("**Skill Gaps**")
                render_skill_chips(item["missing_skills"], is_gap=True)

            with tab_explain:
                st.markdown(
                    (
                        "<div class='candidate-card'>"
                        f"{item['explanation']}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )


def render_multi_role_tab(components: dict[str, Any]) -> None:
    """Render one-resume-to-many-roles matching workflow."""
    loader = components["loader"]
    resume_parser = components["resume_parser"]

    st.subheader("Single Resume → Multiple Roles")
    st.caption(
        "Upload one resume (or pick one from dataset) and discover its best-matching roles.")

    source = st.radio(
        "Resume source",
        options=["Upload One Resume PDF", "Use One Resume from Dataset"],
        horizontal=True,
        key="multi_resume_source",
    )

    candidate_name = ""
    resume_text = ""

    if source == "Upload One Resume PDF":
        uploaded_file = st.file_uploader("Upload resume PDF", type=[
                                         "pdf"], key="multi_resume_upload")
        if uploaded_file is not None:
            candidate_name = file_stem(uploaded_file.name)
            resume_text = resume_parser.extract_text_from_uploaded_bytes(
                uploaded_file.getvalue())
    else:
        resume_limit = st.slider("Resume dataset window", min_value=50,
                                 max_value=1000, value=300, step=50, key="multi_resume_limit")
        resume_records = loader.load_resume_records(nrows=resume_limit)
        labels = [
            f"{item['candidate_id']} | {item.get('category', 'N/A')}" for item in resume_records]
        if labels:
            selected_label = st.selectbox(
                "Select a resume record", labels, key="multi_resume_select")
            selected_record = resume_records[labels.index(selected_label)]
            candidate_name = selected_record["candidate_name"]
            resume_text = selected_record["resume_text"]

    top_k = st.slider("Top roles to return", min_value=3,
                      max_value=20, value=8, step=1, key="multi_top_k")
    job_pool = st.slider("Job search pool size", min_value=200,
                         max_value=5000, value=1200, step=100, key="multi_job_pool")

    run_button = st.button("Find Best Roles", type="primary", key="multi_run")
    if not run_button:
        return

    if not resume_text.strip():
        st.warning("Please provide a valid resume input.")
        return

    with st.spinner("Matching resume to multiple roles..."):
        job_records = loader.load_job_records(nrows=job_pool)
        top_roles = rank_roles_for_resume(
            resume_text=resume_text,
            candidate_name=candidate_name,
            job_records=job_records,
            components=components,
            top_k=top_k,
        )

    if not top_roles:
        st.warning("No role matches were generated.")
        return

    st.markdown(f"### Top role matches for {candidate_name}")
    summary_df = pd.DataFrame(
        [
            {
                "Rank": item["rank"],
                "Role": item["title"],
                "Company": item["company"],
                "Location": item["location"],
                "Work Type": item["work_type"],
                "Match Score (%)": item["score"],
                "Skill Coverage (%)": item["skill_coverage"],
            }
            for item in top_roles
        ]
    )
    st.dataframe(summary_df, width="stretch", hide_index=True)
    st.bar_chart(summary_df.set_index("Role")[
                 ["Match Score (%)", "Skill Coverage (%)"]])

    for item in top_roles:
        with st.expander(f"#{item['rank']} - {item['title']} ({item['score']}%)", expanded=item["rank"] == 1):
            context_bits = [item.get("company", ""), item.get("location", ""), item.get(
                "work_type", ""), item.get("experience_level", "")]
            context_bits = [x for x in context_bits if x]
            if context_bits:
                st.caption(" | ".join(context_bits))

            if item.get("job_industries"):
                st.markdown("**Job Industries**")
                st.write(item["job_industries"])
            if item.get("company_industries"):
                st.markdown("**Company Industries**")
                st.write(item["company_industries"])

            st.markdown("**Matched Skills**")
            st.write(item["matched_skills"] or ["No matched skills found"])
            st.markdown("**Skill Gaps**")
            st.write(item["missing_skills"] or ["No missing skills found"])
            st.markdown("**AI Explanation**")
            st.write(item["explanation"])


def render_dataset_insights_tab(components: dict[str, Any]) -> None:
    """Render dataset health and quality view to improve trust in results."""
    loader = components["loader"]

    st.subheader("Dataset Quality & Insights")
    st.caption(
        "Inspect dataset coverage and cleaned record counts used in matching.")

    col1, col2 = st.columns(2)
    with col1:
        resume_n = st.slider("Resume sample size", min_value=100,
                             max_value=2000, value=600, step=100, key="insights_resume_n")
    with col2:
        job_n = st.slider("Job sample size", min_value=200,
                          max_value=5000, value=1200, step=200, key="insights_job_n")

    if st.button("Refresh Dataset Insights", key="insights_refresh"):
        summary = loader.dataset_quality_summary(
            resume_nrows=resume_n, job_nrows=job_n)

        m1, m2, m3 = st.columns(3)
        m1.metric("Raw Resume Records", summary["raw_resume_records"])
        m1.metric("Resumes with Text", summary["resumes_with_text"])
        m2.metric("Raw Job Records", summary["raw_job_records"])
        m2.metric("Jobs with Description", summary["jobs_with_description"])
        m3.metric("Unique Job IDs", summary["unique_job_ids"])
        m3.metric("Cleaned Job Records", summary["cleaned_job_records"])

        job_records = loader.load_job_records(nrows=job_n)
        if job_records:
            industries = Counter()
            work_types = Counter()
            for item in job_records:
                industries.update(item.get("company_industries", []))
                industries.update(item.get("job_industries", []))
                work_type = item.get("work_type", "")
                if work_type:
                    work_types.update([work_type])

            st.markdown("### Top Industry Signals")
            industry_df = pd.DataFrame(
                [{"Industry": k, "Count": v}
                    for k, v in industries.most_common(12)]
            )
            if industry_df.empty:
                st.info("No industry signals found in current job sample.")
            else:
                st.bar_chart(industry_df.set_index("Industry"))

            st.markdown("### Work Type Distribution")
            work_df = pd.DataFrame(
                [{"Work Type": k, "Count": v}
                    for k, v in work_types.most_common(8)]
            )
            if work_df.empty:
                st.info("No work type data found in current job sample.")
            else:
                st.bar_chart(work_df.set_index("Work Type"))


def main() -> None:
    """Render dashboard UI and execute matching pipeline."""
    st.set_page_config(page_title=DASHBOARD_TITLE, layout="wide")
    apply_page_style()

    st.title(DASHBOARD_TITLE)
    st.markdown('<p class="section-caption">Modular AI screening with embeddings, skill-gap analysis, role ranking, and recruiter insights.</p>', unsafe_allow_html=True)

    components = get_runtime_components()

    tab1, tab2, tab3 = st.tabs(
        ["Candidate Ranking", "Single Resume Multi-Role", "Dataset Insights"])
    with tab1:
        render_candidate_screening_tab(components)
    with tab2:
        render_multi_role_tab(components)
    with tab3:
        render_dataset_insights_tab(components)


if __name__ == "__main__":
    main()
