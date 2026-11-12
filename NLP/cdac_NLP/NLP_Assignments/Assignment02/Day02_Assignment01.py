
"""
Title: Spell Checker using NLP Techniques
Subject: Natural Language Processing

Objective:
To design and implement a spell correction
system using TextBlob and PySpellChecker.
"""

# ==========================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ==========================================

from textblob import TextBlob
from spellchecker import SpellChecker


# ==========================================
# STEP 2: INITIALIZE SPELL CHECKER
# ==========================================

spell = SpellChecker()


# ==========================================
# STEP 3: SPELL CHECKING USING TEXTBLOB
# ==========================================

def correct_using_textblob(text):

    blob = TextBlob(text)

    # Correct spelling of the complete text
    corrected_text = str(blob.correct())

    return corrected_text


# ==========================================
# STEP 4: SPELL CHECKING USING PYSPELLCHECKER
# ==========================================

def correct_using_spellchecker(text):

    words = text.split()

    corrected_words = []

    for word in words:

        # Separate punctuation from word
        clean_word = word.strip(".,!?;:")

        # Identify misspelled word
        correction = spell.correction(clean_word.lower())

        # Preserve original word if no correction
        if correction is None:
            correction = clean_word

        # Preserve uppercase formatting
        if clean_word.istitle():
            correction = correction.capitalize()

        elif clean_word.isupper():
            correction = correction.upper()

        corrected_words.append(correction)

    return " ".join(corrected_words)


# ==========================================
# STEP 5: MAIN PROGRAM
# ==========================================

print("=" * 60)
print("              NLP SPELL CHECKER")
print("=" * 60)

# Take input from user
text = input("\nEnter a sentence with spelling mistakes: ")

# Apply TextBlob
textblob_result = correct_using_textblob(text)

# Apply PySpellChecker
spellchecker_result = correct_using_spellchecker(text)

# Display results
print("\nOriginal Text:")
print(text)

print("\nTextBlob Correction:")
print(textblob_result)

print("\nPySpellChecker Correction:")
print(spellchecker_result)

print("\n" + "=" * 60)
print("          SPELL CHECKING COMPLETED")
print("=" * 60)
