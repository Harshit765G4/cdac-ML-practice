
"""
Title: Grammar Checker using NLP
Subject: Natural Language Processing

Objective:
To develop a grammar checking application
using LanguageTool and spaCy.
"""

# ==========================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ==========================================

import language_tool_python
import spacy


# ==========================================
# STEP 2: INITIALIZE LANGUAGE TOOL
# ==========================================

tool = language_tool_python.LanguageTool('en-US')


# ==========================================
# STEP 3: LOAD SPACY MODEL
# ==========================================

nlp = spacy.load("en_core_web_sm")


# ==========================================
# STEP 4: GRAMMAR ERROR DETECTION
# ==========================================

def check_grammar(text):

    # Detect grammatical errors
    matches = tool.check(text)

    return matches


# ==========================================
# STEP 5: GRAMMAR CORRECTION
# ==========================================

def correct_grammar(text):

    matches = check_grammar(text)

    # Apply suggested corrections
    corrected_text = language_tool_python.utils.correct(
        text, matches
    )

    return corrected_text


# ==========================================
# STEP 6: SPACY LINGUISTIC ANALYSIS
# ==========================================

def analyze_text(text):

    doc = nlp(text)

    print("\nSPACY LINGUISTIC ANALYSIS")
    print("-" * 50)

    for token in doc:

        print(
            f"Word: {token.text:<15} "
            f"POS: {token.pos_:<10} "
            f"Lemma: {token.lemma_}"
        )


# ==========================================
# STEP 7: MAIN PROGRAM
# ==========================================

print("=" * 60)
print("              NLP GRAMMAR CHECKER")
print("=" * 60)

# Take input from user
text = input("\nEnter a sentence with grammatical mistakes: ")

# Detect errors
matches = check_grammar(text)

print("\nORIGINAL TEXT:")
print(text)

# Display detected errors
print("\nDETECTED GRAMMAR ERRORS:")
print("-" * 50)

if matches:

    for index, match in enumerate(matches, start=1):

        print(f"\nError {index}:")
        print("Message:", match.message)
        print("Incorrect Text:", text[match.offset:match.offset + match.error_length])
        print("Suggestions:", match.replacements[:5])

else:

    print("No grammatical errors detected.")

# Correct grammar
corrected_text = language_tool_python.utils.correct(
    text, matches
)

print("\nCORRECTED TEXT:")
print(corrected_text)

# Perform spaCy analysis
analyze_text(text)

print("\n" + "=" * 60)
print("         GRAMMAR CHECKING COMPLETED")
print("=" * 60)

# Close LanguageTool server
tool.close()
