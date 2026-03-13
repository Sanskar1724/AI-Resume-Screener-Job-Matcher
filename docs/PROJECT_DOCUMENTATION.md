# Project Documentation: AI Resume Screener & Job Matcher

## 1. Overview

AI Resume Screener & Job Matcher is a modular Python application for recruiter-side resume analysis and role matching. It combines resume parsing, NLP-based skill/entity extraction, sentence embeddings, skill-gap analysis, and interactive visualization through Streamlit.

The system supports three practical workflows:

- many candidates matched against one target role
- one resume matched against many roles
- dataset quality inspection for better trust in results

The project is designed around real datasets already available in the workspace and does not generate synthetic data.

---

## 2. Goals

Primary goals of the project:

- reduce manual screening effort
- improve resume-job fit analysis beyond simple keyword matching
- identify missing skills clearly
- generate recruiter-friendly explanations
- provide an interactive dashboard for analysis and decision support

---

## 3. Core Capabilities

### Resume Processing

- extracts text from PDF resumes using `pdfplumber`
- falls back to `PyMuPDF` when needed
- normalizes noisy text before downstream processing

### NLP Extraction

- skill extraction using a controlled skills vocabulary from dataset
- entity extraction for education, organizations, locations, and experience mentions
- heuristic fallback for environments where `spaCy` model execution is limited

### Embedding-Based Matching

- converts resumes and job descriptions into dense semantic vectors
- uses `sentence-transformers/all-MiniLM-L6-v2`
- compares semantic meaning, not only literal keyword overlap

### Matching and Ranking

- computes similarity score from vector comparison
- computes skill coverage and missing skills
- ranks candidates by descending score
- supports multi-role recommendation for a single resume

### Recruiter Dashboard

- candidate ranking workflow
- single resume to multiple roles workflow
- dataset quality dashboard
- visual skill insights and recruiter-readable explanations

---

## 4. Architecture

## High-Level Flow

1. Load datasets
2. Parse resume text or read resume dataset text
3. Extract skills and entities
4. Build embedding vectors
5. Compare against job descriptions
6. Compute similarity and skill-gap metrics
7. Rank and explain results
8. Present results in Streamlit

## Technical Stack Summary

- **Language:** Python 3.11
- **Frontend:** Streamlit
- **Data processing:** pandas, numpy
- **Machine learning utilities:** scikit-learn
- **Semantic matching:** sentence-transformers using `all-MiniLM-L6-v2`
- **NLP:** spaCy with heuristic fallback for unsupported environments
- **Document parsing:** pdfplumber and PyMuPDF
- **Deployment target:** Streamlit Cloud

## Module Responsibilities

### [config.py](../config.py)

Central location for file paths, model names, and application constants.

### [data_loader/loader.py](../data_loader/loader.py)

Loads and enriches datasets:

- skills vocabulary
- job postings
- job skills mapping
- job industries mapping
- company metadata
- resume dataset

It also builds cleaned job records and dataset quality summaries.

### [resume_parser/parser.py](../resume_parser/parser.py)

Handles PDF extraction for uploaded resumes.

### [nlp/skill_extractor.py](../nlp/skill_extractor.py)

Matches normalized n-grams against the skills database.

### [nlp/entity_extractor.py](../nlp/entity_extractor.py)

Extracts resume metadata using `spaCy` when available and heuristic fallback otherwise.

### [embeddings/embedding_model.py](../embeddings/embedding_model.py)

Wraps `SentenceTransformer` model loading and vector generation.

### [matching/similarity.py](../matching/similarity.py)

Computes vector similarity score.

### [matching/skill_gap.py](../matching/skill_gap.py)

Computes matched and missing skills plus skill coverage.

### [matching/ranking.py](../matching/ranking.py)

Sorts candidates or role matches by score and assigns rank.

### [ai_insights/explanation.py](../ai_insights/explanation.py)

Builds recruiter-facing explanation text using score, skills, and extracted entities.

### [dashboard/streamlit_app.py](../dashboard/streamlit_app.py)

Main UI entry point. Contains:

- dashboard styling
- candidate screening workflow
- multi-role workflow
- dataset insights workflow
- visual result presentation

### [app.py](../app.py)

CLI entry point for quick testing and dataset-based ranking from terminal.

### [utils/helpers.py](../utils/helpers.py)

Shared normalization and utility helpers.

---

## 5. Folder Structure

```text
project/
├── .streamlit/
│   └── config.toml
├── .gitignore
├── README.md
├── app.py
├── config.py
├── runtime.txt
├── packages.txt
├── requirements.txt
├── datasets/
│   ├── LinkedIn Job Dataset/
│   ├── Resume/
│   └── Skills Dataset/
├── ai_insights/
│   └── explanation.py
├── dashboard/
│   └── streamlit_app.py
├── data_loader/
│   └── loader.py
├── docs/
│   └── PROJECT_DOCUMENTATION.md
├── embeddings/
│   └── embedding_model.py
├── matching/
│   ├── ranking.py
│   ├── similarity.py
│   └── skill_gap.py
├── nlp/
│   ├── entity_extractor.py
│   └── skill_extractor.py
├── resume_parser/
│   └── parser.py
└── utils/
    └── helpers.py
```

---

## 6. Datasets Used

The application relies on datasets stored inside the repository at `project/datasets/` so the same code works both locally and on Streamlit Cloud.

### Skills Dataset

Used for controlled vocabulary skill matching.

- `technical_skills.csv`

### Resume Dataset

Used for dataset-driven candidate screening.

- `Resume.csv`

### LinkedIn Job Dataset

Used for job descriptions and job metadata.

- `job_postings.csv`
- `job_skills.csv`
- `job_industries.csv`
- `companies.csv`
- `company_industries.csv`

## Dataset Enrichment Strategy

The loader enriches job records by combining:

- job description text
- skill tags from job-skill mapping
- industry metadata from job/company tables
- company names and contextual metadata

This produces a richer `matching_text` field for more accurate semantic comparison.

## Current Dataset Packaging Strategy

To keep the repository deployable on GitHub and Streamlit Cloud:

- very large raw datasets were not referenced from outside the repo anymore
- sampled deployable CSVs were added inside `project/datasets/`
- `job_postings.csv` was reduced to 3000 rows for cloud-friendly usage
- `Resume.csv` was reduced to 1000 rows for cloud-friendly usage
- smaller mapping datasets were included as-is

---

## 7. Matching Logic

## Candidate Screening

Input:

- multiple resumes
- one target job description

Output:

- ranked candidate list
- overall match score
- skill coverage
- missing skills
- AI explanation

## Single Resume Multi-Role

Input:

- one resume
- multiple job records

Output:

- top matching roles
- role rank and score
- role-specific skill gap analysis
- recruiter summary for each role

## Scoring Concept

The project combines:

- semantic alignment from sentence embeddings
- explicit skill overlap from controlled vocabulary
- explainability from extracted resume signals

## End-to-End Working Logic

### Step 1: Data ingestion

`DatasetLoader` reads the jobs dataset, job-skill mappings, industry mappings, company metadata, and resume dataset.

### Step 2: Job record enrichment

Each job posting is enriched by joining:

- job title
- job description
- skill tags
- job industries
- company industries
- company name

These fields are merged into a stronger `matching_text` representation for semantic comparison.

### Step 3: Resume parsing

Uploaded PDF resumes are parsed into text using `pdfplumber`, with fallback support through `PyMuPDF`.

### Step 4: Skill and entity extraction

The system extracts:

- explicit technical skills from the skill vocabulary
- education / organization / location / experience signals through entity extraction

### Step 5: Embedding generation

Both resume text and job `matching_text` are converted into dense vectors using the Sentence Transformer model.

### Step 6: Scoring and ranking

The application compares embeddings, computes similarity scores, analyzes skill overlap, and ranks the best candidates or roles.

### Step 7: Explainable output

The dashboard shows score, matched skills, missing skills, skill coverage, charts, and recruiter-readable explanations.

Displayed metrics include:

- match score (%)
- skill coverage (%)
- matched skills
- missing skills

---

## 8. Streamlit Workflows

## Candidate Ranking Tab

Purpose:
Compare multiple candidates against one role.

Includes:

- source selection for candidates and jobs
- ranking table
- top candidate summary cards
- score analysis charts
- matched skill and gap charts
- structured candidate detail view

## Single Resume Multi-Role Tab

Purpose:
Identify the best roles for one candidate.

Includes:

- resume upload or dataset selection
- top role ranking table
- role comparison chart
- detailed per-role skill analysis

## Dataset Insights Tab

Purpose:
Show input data quality and distribution summaries.

Includes:

- record counts
- cleaned job counts
- coverage metrics
- industry distribution
- work-type distribution

---

## 9. Setup Instructions

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

## Run CLI

```bash
python app.py --dataset-mode --dataset-limit 120 --job "Python, SQL, machine learning, NLP"
```

---

## 10. Environment Notes

The current implementation runs in the workspace environment, but there is one practical compatibility note:

- `spaCy` has issues on Python `3.14`
- the project already handles this by falling back to heuristics in entity extraction
- for best long-term stability, Python `3.11` or `3.12` is recommended

Additional runtime notes:

- the first launch may download the sentence-transformer model from Hugging Face
- Windows may show symlink warnings during model caching; they are usually non-fatal

## Deployment Notes

The project is prepared for Streamlit Cloud with the following files:

- [../.streamlit/config.toml](../.streamlit/config.toml) for Streamlit settings and theme
- [../runtime.txt](../runtime.txt) to pin Python 3.11
- [../requirements.txt](../requirements.txt) for Python dependencies
- [../packages.txt](../packages.txt), which is intentionally empty because previous Linux system package entries caused installation failures on Streamlit Cloud

Important deployment choices already implemented:

- datasets are bundled inside the repository
- `config.py` resolves data from `ROOT_DIR / "datasets"`
- local imports in `streamlit_app.py` are placed below the `sys.path` bootstrap for deployment-safe imports

---

## 11. Accuracy Considerations

Accuracy depends on:

- quality of job descriptions
- consistency of resume text extraction
- quality and coverage of the skills dataset
- normalization of abbreviations and aliases

Current accuracy improvements already included:

- dataset joins across jobs, skills, industries, and companies
- cleaned job record generation
- duplicate filtering
- low-quality description filtering
- enriched job matching text

Recommended next improvements:

- alias mapping for common skill abbreviations
- domain-specific weighting for high-value skills
- hybrid scoring formula combining embedding score and explicit skill overlap
- resume section parsing for better education/experience extraction
- optional LLM summarization for more polished explanations

---

## 12. Known Limitations

- skill extraction is vocabulary-based, so unseen aliases may be missed
- resume entity extraction fallback is simpler when `spaCy` is unavailable
- job-role accuracy depends on dataset text quality
- current ranking is deterministic and not learned from labeled hiring outcomes
- very large dataset matching may need batch optimization for speed

---

## 13. Suggested Future Enhancements

- add batch embedding cache for faster repeated dashboard runs
- add downloadable CSV/PDF recruiter reports
- add advanced filters by category, industry, location, and experience level
- add candidate comparison view between top 2 or top 3 applicants
- add fine-tuned domain models for specific hiring verticals
- add REST API layer for integration with external ATS tools
- add tests and CI workflow for automated validation

---

## 14. Example Use Cases

### Recruiter Workflow

1. Select target job from dataset
2. Load resumes from dataset or uploaded PDFs
3. Run ranking
4. Review top candidates and skill gaps
5. Shortlist based on high match + manageable gaps

### Candidate Career Guidance Workflow

1. Upload one resume
2. Run multi-role matching
3. Review best matching roles
4. Inspect repeated skill gaps
5. Identify learning priorities

---

## 15. Publication Notes for GitHub

Recommended repository extras:

- MIT License
- screenshots of dashboard tabs
- architecture diagram
- sample output screenshots
- CI workflow for lint/import validation

Suggested repository subtitle:
AI-powered resume screening and job matching system using NLP, embeddings, skill-gap analysis, and Streamlit.

---

## 16. Conclusion

This project is a practical AI recruiting application built with modular Python components and real datasets. It goes beyond plain keyword filtering by combining semantic similarity, structured skill analysis, and recruiter-friendly presentation.

It is suitable as:

- a portfolio project
- a datathon or hackathon submission
- a base system for ATS or recruiting workflow extensions
