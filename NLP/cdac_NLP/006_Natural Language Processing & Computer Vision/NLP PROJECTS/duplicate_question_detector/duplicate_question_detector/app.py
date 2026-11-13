"""Duplicate Question Detection using sentence embeddings and cosine similarity.

Run from this folder with:
    python app.py
"""

from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.70

# Small classroom dataset. Add more reference questions as needed.
EXISTING_QUESTIONS = {
    1: "How to sort a list in Python?",
    2: "What is the difference between list.sort() and sorted() in Python?",
    3: "How to concatenate strings in Python?",
    4: "How can I merge multiple lists in Python?",
    5: "How do I install a Python package using pip?",
    6: "What is a dictionary in Python and how is it used?",
    7: "How can I read a CSV file using pandas?",
    8: "How do I create a function in Python?",
}


def load_model():
    """Load the sentence-embedding model once."""
    print(f"Loading sentence embedding model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def build_reference_embeddings(model, questions):
    """Encode all reference questions in one batch."""
    question_ids = list(questions.keys())
    question_texts = [questions[q_id] for q_id in question_ids]
    embeddings = model.encode(
        question_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return question_ids, embeddings


def find_duplicates(model, question_ids, reference_embeddings, new_question,
                    threshold=DEFAULT_THRESHOLD, top_k=5):
    """Return reference questions whose cosine similarity meets the threshold."""
    if not new_question or not new_question.strip():
        raise ValueError("Please enter a question before searching.")

    query_embedding = model.encode(
        new_question.strip(), convert_to_tensor=True, normalize_embeddings=True
    )
    similarities = util.cos_sim(query_embedding, reference_embeddings)[0]
    ranked_indices = similarities.argsort(descending=True).tolist()

    matches = []
    for index in ranked_indices:
        score = float(similarities[index].item())
        if score >= threshold:
            q_id = question_ids[index]
            matches.append({
                "question_id": q_id,
                "question_text": EXISTING_QUESTIONS[q_id],
                "similarity": round(score, 4),
            })
        if len(matches) >= top_k:
            break
    return matches


def main():
    model = load_model()
    question_ids, reference_embeddings = build_reference_embeddings(
        model, EXISTING_QUESTIONS
    )
    print("\nReference questions and embeddings are ready.")
    print(f"Number of reference questions: {len(question_ids)}")
    print("\nType a new question. Type 'exit' to stop.\n")

    while True:
        query = input("Enter a question: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Duplicate Question Detector closed.")
            break
        if not query:
            print("Please enter a non-empty question.\n")
            continue

        try:
            matches = find_duplicates(
                model, question_ids, reference_embeddings, query,
                threshold=DEFAULT_THRESHOLD,
            )
            if matches:
                print("\nPossible duplicate / similar questions:")
                for match in matches:
                    print(
                        f"  ID {match['question_id']} | "
                        f"Similarity: {match['similarity']:.4f} | "
                        f"{match['question_text']}"
                    )
            else:
                print("\nNo question reached the similarity threshold.")
            print()
        except ValueError as exc:
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    main()
