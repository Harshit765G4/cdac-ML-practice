
"""
Streamlit demo for resume parsing and job-skill matching.

Run from this folder with:
streamlit run app.py
"""

import json
from pathlib import Path

with open("candidates.json", "r", encoding="utf-8") as file:
    SAMPLE_CANDIDATES = json.load(file)

import pandas as pd
import streamlit as st

from parser import match_candidates, parse_resume, extract_skills


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Resume Parser and Candidate Matcher",
    page_icon="📄",
    layout="wide"
)

st.title("Resume Parser & Candidate Matcher")

st.write(
    "Extract candidate details with spaCy and regular expressions, "
    "then rank candidates by required-skill coverage."
)


# --------------------------------------------------
# Sample Job Description
# --------------------------------------------------

SAMPLE_JOB = """
We are looking for a Python developer with experience in machine learning,
natural language processing, SQL, pandas, and scikit-learn.
Experience with FastAPI and Docker is an advantage.
"""


# --------------------------------------------------
# Sample Candidate Resumes
# --------------------------------------------------

candidate_pairs = [
    (c["candidate_id"], c["resume_text"])
    for c in SAMPLE_CANDIDATES
]

# --------------------------------------------------
# Job Description Input
# --------------------------------------------------

job_description = st.text_area(
    "Job description",
    value=SAMPLE_JOB,
    height=130
)


# --------------------------------------------------
# Resume Upload
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload candidate resumes as .txt files (optional). "
    "When none are uploaded, the three sample resumes are used.",
    type=["txt"],
    accept_multiple_files=True
)


# --------------------------------------------------
# Prepare Candidate Data
# --------------------------------------------------

candidate_pairs = []

if uploaded_files:

    for file in uploaded_files:

        try:
            content = file.read().decode("utf-8-sig")

        except UnicodeDecodeError:
            st.error(
                f"Could not read {file.name}. "
                "Save the file as UTF-8 text and upload again."
            )
            continue

        candidate_pairs.append((file.name, content))

else:

    candidate_pairs = [
        (candidate["candidate_id"], candidate["resume_text"])
        for candidate in SAMPLE_CANDIDATES
    ]

    st.info("Using built-in sample resumes for the first demonstration.")


# --------------------------------------------------
# Parse and Rank Candidates
# --------------------------------------------------

if st.button("Parse resumes and rank candidates", type="primary"):

    try:

        # Extract skills from job description
        required_skills = extract_skills(job_description)

        if not required_skills:

            st.error(
                "No configured skills were detected in the job description. "
                "Add terms such as Python, SQL, or machine learning."
            )

        else:

            # Display required skills
            st.subheader("Required skills detected")

            st.write(", ".join(required_skills))

            # Match candidates against job requirements
            ranked = match_candidates(
                job_description,
                candidate_pairs
            )

            # Prepare ranking table
            table_rows = [
                {
                    "Rank": index + 1,
                    "Candidate": row["candidate_id"],
                    "Detected name": row["candidate_name"],
                    "Match score": f"{row['match_score'] * 100:.1f}%",
                    "Matched skills": ", ".join(row["matched_skills"]),
                    "Missing skills": ", ".join(row["missing_skills"]),
                }
                for index, row in enumerate(ranked)
            ]

            # Display candidate ranking
            st.subheader("Candidate ranking")

            st.dataframe(
                pd.DataFrame(table_rows),
                use_container_width=True,
                hide_index=True
            )

            # Display extracted information
            st.subheader("Extracted resume information")

            for row in ranked:

                with st.expander(
                    f"{row['candidate_id']} — "
                    f"{row['candidate_name']} "
                    f"({row['match_score'] * 100:.1f}%)"
                ):

                    st.json(row["parsed_information"])

    except Exception as exc:

        st.error(str(exc))

        st.info(
            "Check that the spaCy model is installed "
            "in the active nlp-env environment."
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Teaching note: This is a rule-assisted prototype. "
    "spaCy identifies general entities; the editable vocabulary "
    "identifies skills. It is not a validated hiring or "
    "employment decision system."
)
