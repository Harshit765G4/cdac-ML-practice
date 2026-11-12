
"""
Title: Rule-Based Chatbot Using Python
Subject: Natural Language Processing

Objective:
To develop a simple rule-based chatbot
using NLU and NLG concepts.
"""

import random
import re


# ==========================================
# STEP 1: KNOWLEDGE BASE
# ==========================================

responses = {

    "greeting": [
        "Hello! How can I help you?",
        "Hi! Nice to meet you.",
        "Hey! What can I do for you?"
    ],

    "goodbye": [
        "Goodbye! Have a great day.",
        "See you again!",
        "Bye! Take care."
    ],

    "name": [
        "I am a Python-based chatbot.",
        "My name is NLP Bot."
    ],

    "help": [
        "I can answer basic questions.",
        "You can ask me about my name, time, or greetings."
    ],

    "thanks": [
        "You're welcome!",
        "Happy to help!",
        "No problem!"
    ],

    "time": [
        "You can check the current time on your device."
    ]
}


# ==========================================
# STEP 2: TEXT PREPROCESSING
# ==========================================

def preprocess_text(text):

    # Convert text into lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Tokenization
    words = text.split()

    return words


# ==========================================
# STEP 3: NATURAL LANGUAGE UNDERSTANDING
# ==========================================


def understand_intent(words):

    word_set = set(words)

    # Check more specific intents first

    if "time" in word_set:
        return "time"

    elif word_set.intersection({"goodbye", "bye", "exit", "quit"}):
        return "goodbye"

    elif word_set.intersection({"thanks", "thank"}):
        return "thanks"

    elif word_set.intersection({"help", "support"}):
        return "help"

    elif "name" in word_set:
        return "name"

    elif word_set.intersection({"hello", "hi", "hey", "goodmorning"}):
        return "greeting"

    else:
        return "unknown"


# ==========================================
# STEP 4: NATURAL LANGUAGE GENERATION
# ==========================================

def generate_response(intent):

    if intent in responses:

        # Select a random response
        return random.choice(responses[intent])

    else:
        return (
            "Sorry, I didn't understand that. "
            "Could you please rephrase?"
        )


# ==========================================
# STEP 5: MAIN CHATBOT
# ==========================================

def chatbot():

    print("=" * 50)
    print("          WELCOME TO NLP CHATBOT")
    print("=" * 50)

    print("Bot: Hello! I am your chatbot.")
    print("Bot: Type 'bye' to exit.")

    while True:

        # Take user input
        user_input = input("\nYou: ")

        # Preprocessing
        words = preprocess_text(user_input)

        # Identify user intent
        intent = understand_intent(words)

        # Generate response
        response = generate_response(intent)

        # Display chatbot response
        print("Bot:", response)

        # Stop chatbot
        if intent == "goodbye":
            break


# ==========================================
# EXECUTE PROGRAM
# ==========================================

if __name__ == "__main__":
    chatbot()
