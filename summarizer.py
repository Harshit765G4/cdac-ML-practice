
import nltk
import networkx as nx

from functools import lru_cache
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# Download tokenizer resources if missing
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(
            f"tokenizers/{resource}"
        )
    except LookupError:
        nltk.download(resource, quiet=True)


# BART model
MODEL_NAME = "facebook/bart-large-cnn"


@lru_cache(maxsize=1)
def load_bart_model():
    """Load BART only once and reuse it."""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


def abstractive_summary(
    text,
    max_length=150,
    min_length=50,
    num_beams=4
):
    """Generate an abstractive summary using BART."""

    if not text or not text.strip():
        return "Please provide some text."

    tokenizer, model, device = load_bart_model()

    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )


def extractive_summary(text, sentences_count=3):
    """Generate an extractive summary using TextRank."""

    if not text or not text.strip():
        return "Please provide some text."

    sentences = sent_tokenize(text)

    if len(sentences) <= sentences_count:
        return " ".join(sentences)

    # Convert sentences into TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english")

    sentence_vectors = vectorizer.fit_transform(sentences)

    # Calculate sentence similarity
    similarity_matrix = cosine_similarity(sentence_vectors)

    # Build similarity graph
    graph = nx.from_numpy_array(similarity_matrix)

    # Apply PageRank
    scores = nx.pagerank(graph, weight="weight")

    # Select top-ranked sentences
    ranked_sentences = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    selected_indices = sorted(
        index
        for index, score in ranked_sentences[:sentences_count]
    )

    # Preserve original sentence order
    summary = [
        sentences[index]
        for index in selected_indices
    ]

    return " ".join(summary)
