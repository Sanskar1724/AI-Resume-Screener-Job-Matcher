# AI Resume Screener & Job Matcher

Production-style Python project for parsing resumes, extracting skills and entities, generating semantic embeddings, matching candidates to roles, identifying skill gaps, ranking applicants, and presenting recruiter-ready insights through a Streamlit dashboard.

## GitHub Description (Short)

AI-powered resume screening and job matching system using NLP, sentence embeddings, skill-gap analysis, and an interactive Streamlit recruiter dashboard.

## Detailed Documentation

For full technical documentation, architecture, workflows, dataset usage, deployment notes, and future improvements, see [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md).

## Tech Stack

- Python 3.11
- Streamlit
- pandas / numpy
- scikit-learn
- sentence-transformers (`all-MiniLM-L6-v2`)
- spaCy with graceful fallback
- pdfplumber / PyMuPDF

## Key Features

- Parse resume PDFs with robust text extraction (`pdfplumber` + fallback)
- Extract technical skills using a skills database
- Extract entities (education, experience, organizations, locations)
- Generate semantic embeddings with `sentence-transformers`
- Compute resume-job similarity using cosine-style vector scoring
- Perform skill-gap analysis (matched vs missing skills)
- Rank candidates by overall fit score
- Generate AI-readable recruiter explanations
- Match one resume against multiple roles
- Provide dataset quality and insights view for better trust in results

## Project Structure

```
project/
│
├── .streamlit/
│   └── config.toml
├── app.py
├── config.py
├── runtime.txt
├── packages.txt
├── requirements.txt
├── README.md
├── datasets/
│   ├── LinkedIn Job Dataset/
│   ├── Resume/
│   └── Skills Dataset/
│
├── data_loader/
│   └── loader.py
│
├── resume_parser/
│   └── parser.py
│
├── nlp/
│   ├── skill_extractor.py
│   └── entity_extractor.py
│
├── embeddings/
│   └── embedding_model.py
│
├── matching/
│   ├── similarity.py
│   ├── ranking.py
│   └── skill_gap.py
│
├── ai_insights/
│   └── explanation.py
│
├── dashboard/
│   └── streamlit_app.py
│
└── utils/
    └── helpers.py
```

## Dataset Expectations

This project is configured to use the bundled `project/datasets/` directory so it can run on Streamlit Cloud without depending on files outside the repository.

- Resume dataset (text resumes)
- LinkedIn job postings and skills mappings
- Skills vocabulary dataset
- Company and industry mapping datasets

No synthetic data generation is required.

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

## Deployment Notes

- Streamlit Cloud entry file: `dashboard/streamlit_app.py`
- Streamlit config: `.streamlit/config.toml`
- Python runtime pin: `runtime.txt`
- Dataset paths are resolved from `config.py` using `ROOT_DIR / "datasets"`
- `packages.txt` is intentionally empty because the earlier Linux package workaround caused dependency failures on Streamlit Cloud

### 3) Optional CLI run

```bash
python app.py --dataset-mode --dataset-limit 120 --job "Data Scientist role requiring Python, SQL, ML, and NLP."
```

## Dashboard Workflows

### Candidate Ranking

Compare many resumes against one target role (manual JD or dataset role).

### Single Resume Multi-Role

Upload/select one resume and discover top-matching roles from the jobs dataset.

### Dataset Insights

View quality metrics and distribution insights to improve model trust and analysis quality.

## Why This Project

Traditional resume filtering is keyword-heavy and brittle. This project adds semantic matching + skill gap intelligence + explainable ranking to support faster and more consistent hiring decisions.

## Suggested GitHub Topics

`python` `nlp` `resume-screening` `job-matching` `streamlit` `sentence-transformers` `recruitment-ai` `skill-gap-analysis`

## License

Add your preferred license (MIT recommended for open-source demo projects).
