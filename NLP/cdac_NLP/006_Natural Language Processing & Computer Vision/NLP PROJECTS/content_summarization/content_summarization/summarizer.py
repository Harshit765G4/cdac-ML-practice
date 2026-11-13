"""Abstractive BART and extractive TextRank summarization utilities."""

from functools import lru_cache

MODEL_NAME = "facebook/bart-large-cnn"


@lru_cache(maxsize=1)
def load_bart_model():
    """Load and cache BART so Streamlit does not reload it on every rerun."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def abstractive_summary(article, max_length=150, min_length=50, num_beams=4):
    """Generate a new summary with BART. Inputs are truncated at 1024 tokens."""
    if not article or not article.strip():
        raise ValueError("Please provide article text.")

    import torch

    tokenizer, model, device = load_bart_model()
    encoded = tokenizer(
        article.strip(),
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )
    input_token_count = int(encoded["input_ids"].shape[1])
    encoded = {key: value.to(device) for key, value in encoded.items()}

    # Keep the requested PDF defaults for normal-length articles, while
    # reducing limits for short inputs so generation does not demand more
    # summary tokens than the source can reasonably support.
    safe_max = min(max_length, max(20, input_token_count - 2))
    safe_min = min(min_length, max(5, safe_max // 2))
    if safe_min >= safe_max:
        safe_min = max(1, safe_max - 1)

    with torch.no_grad():
        summary_ids = model.generate(
            **encoded,
            max_length=safe_max,
            min_length=safe_min,
            do_sample=False,
            num_beams=num_beams,
            early_stopping=True,
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def extractive_summary(article, sentences_count=3):
    """Select important original sentences using Sumy's TextRank."""
    if not article or not article.strip():
        raise ValueError("Please provide article text.")

    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer

    parser = PlaintextParser.from_string(article.strip(), Tokenizer("english"))
    summarizer = TextRankSummarizer()
    selected_sentences = summarizer(parser.document, sentences_count=sentences_count)
    return " ".join(str(sentence) for sentence in selected_sentences)
