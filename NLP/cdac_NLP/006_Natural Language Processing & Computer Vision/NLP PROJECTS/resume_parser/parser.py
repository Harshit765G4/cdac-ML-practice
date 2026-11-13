
"""
Resume information extraction and candidate-to-job skill matching.

Default spaCy model: en_core_web_sm

To use the transformer model:
Set SPACY_MODEL=en_core_web_trf
"""

import os
import re
from functools import lru_cache


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")


# --------------------------------------------------
# Skill Vocabulary
# --------------------------------------------------

SKILL_ALIASES = {
    "python": ["python"],
    "java": ["java"],
    "c++": ["c++"],
    "c#": ["c#"],
    "sql": ["sql"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning"],
    "natural language processing": ["natural language processing", "nlp"],
    "computer vision": ["computer vision"],
    "data analysis": ["data analysis", "data analytics"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow"],
    "spacy": ["spacy"],
    "transformers": ["transformers", "hugging face"],
    "postgresql": ["postgresql", "postgres"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb"],
    "fastapi": ["fastapi"],
    "flask": ["flask"],
    "django": ["django"],
    "html": ["html"],
    "css": ["css"],
    "javascript": ["javascript"],
    "react": ["react"],
    "git": ["git"],
    "docker": ["docker"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "power bi": ["power bi"],
    "tableau": ["tableau"],
    "opencv": ["opencv", "open cv"],
}


# --------------------------------------------------
# Education Vocabulary
# --------------------------------------------------

EDUCATION_TERMS = [
    "b.e.", "b.e", "b.tech", "btech",
    "m.e.", "m.e", "m.tech", "mtech",
    "b.sc", "bsc", "m.sc", "msc",
    "mba", "ph.d", "phd",
    "bachelor", "master", "diploma",
]


# --------------------------------------------------
# Load spaCy Model
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_nlp():

    import spacy

    try:
        return spacy.load(SPACY_MODEL)

    except OSError as exc:

        raise RuntimeError(
            f"spaCy model '{SPACY_MODEL}' is not installed. Run: "
            f"python -m spacy download {SPACY_MODEL}"
        ) from exc


# --------------------------------------------------
# Phrase Matching
# --------------------------------------------------

def _contains_phrase(text_lower, phrase):
    """
    Match a phrase without matching it inside a longer alphanumeric word.
    """

    escaped = re.escape(phrase.lower())

    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"

    return re.search(pattern, text_lower) is not None


# --------------------------------------------------
# Extract Skills
# --------------------------------------------------

def extract_skills(text):

    text_lower = text.lower()

    found = []

    for canonical_skill, aliases in SKILL_ALIASES.items():

        if any(
            _contains_phrase(text_lower, alias)
            for alias in aliases
        ):

            found.append(canonical_skill)

    return sorted(found)


# --------------------------------------------------
# Parse Resume
# --------------------------------------------------

def parse_resume(resume_text):

    if not resume_text or not resume_text.strip():

        raise ValueError("Resume text cannot be blank.")

    nlp = load_nlp()

    doc = nlp(resume_text)

    names = []
    organizations = []

    # Extract named entities using spaCy
    for entity in doc.ents:

        if entity.label_ == "PERSON" and entity.text not in names:

            names.append(entity.text)

        elif entity.label_ == "ORG" and entity.text not in organizations:

            organizations.append(entity.text)

    # Extract email addresses
    emails = sorted(
        set(
            re.findall(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                resume_text
            )
        )
    )

    # Extract phone numbers
    phone_candidates = re.findall(
        r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)",
        resume_text
    )

    phones = sorted({
        re.sub(r"\s+", " ", value).strip()
        for value in phone_candidates
    })

    # Extract education
    lower_text = resume_text.lower()

    education = [
        term
        for term in EDUCATION_TERMS
        if _contains_phrase(lower_text, term)
    ]

    return {
        "names": names,
        "emails": emails,
        "phones": phones,
        "skills": extract_skills(resume_text),
        "experiences": organizations,
        "education": education,
    }


# --------------------------------------------------
# Match Candidates Against Job Description
# --------------------------------------------------

def match_candidates(job_description, candidate_resumes):
    """
    Rank (candidate_id, resume_text) pairs by required-skill coverage.
    """

    required_skills = extract_skills(job_description)

    if not required_skills:

        raise ValueError(
            "No skills from the configured skill vocabulary were found "
            "in the job description. Add the required terms to "
            "SKILL_ALIASES or edit the job description."
        )

    results = []

    for candidate_id, resume_text in candidate_resumes:

        parsed = parse_resume(resume_text)

        # Find common skills
        matched = sorted(
            set(parsed["skills"]) & set(required_skills)
        )

        # Calculate matching score
        score = len(matched) / len(required_skills)

        results.append({

            "candidate_id": candidate_id,

            "candidate_name": (
                parsed["names"][0]
                if parsed["names"]
                else "Not detected"
            ),

            "match_score": round(score, 4),

            "matched_skills": matched,

            "missing_skills": sorted(
                set(required_skills) - set(matched)
            ),

            "parsed_information": parsed,

        })

    # Sort candidates by highest matching score
    return sorted(
        results,
        key=lambda item: item["match_score"],
        reverse=True
    )
