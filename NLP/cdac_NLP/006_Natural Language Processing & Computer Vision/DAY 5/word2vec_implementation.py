# !pip install gensim

# IMPLEMENTATION OF WORD EMBEDDINGS USING WORD2VEC

import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# STEP 1: Load dataset
data = pd.read_csv(r"C:\Users\Hey_Nerd\Ml\cdac_ML_21_oct_2026\NLP\cdac_NLP\006_Natural Language Processing & Computer Vision\DAY 5\sentiment_dataset.csv")

reviews = data["Review"].fillna("").astype(str)

# STEP 2: Text preprocessing and tokenization
sentences = [
    simple_preprocess(review, deacc=True, min_len=1)
    for review in reviews
]

print("First tokenized sentence:")
print(sentences[0])

# STEP 3: Demonstrate One-Hot Encoding

sample_sentence = "the product is excellent"

words = sample_sentence.split()

vocabulary = sorted(set(words))

print("\nVocabulary:")
print(vocabulary)

print("\nOne-Hot Encoding:")

for word in vocabulary:

    vector = np.zeros(len(vocabulary), dtype=int)

    index = vocabulary.index(word)

    vector[index] = 1

    print(word, ":", vector)

# STEP 4: Train CBOW Model
print("\nTRAINING CBOW MODEL")

cbow_model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=3,
    min_count=1,
    workers=1,
    sg=0,
    epochs=100,
    seed=42
)

print("CBOW training completed!")

# STEP 5: Train Skip-gram Model
print("\nTRAINING SKIP-GRAM MODEL")

skipgram_model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=3,
    min_count=1,
    workers=1,
    sg=1,
    epochs=100,
    seed=42
)

print("Skip-gram training completed!")

# STEP 6: Display Word Vectors

word = "excellent"

print("\nWORD VECTOR FOR:", word)

print("\nCBOW Vector:")
print(cbow_model.wv[word])

print("\nSkip-gram Vector:")
print(skipgram_model.wv[word])

print("\nVector Dimensions:")
print("CBOW:", cbow_model.wv[word].shape)
print("Skip-gram:", skipgram_model.wv[word].shape)

# STEP 7: Find Similar Words

print("\nCBOW - MOST SIMILAR WORDS")

print(cbow_model.wv.most_similar(
    "excellent", topn=5
))

print("\nSKIP-GRAM - MOST SIMILAR WORDS")

print(skipgram_model.wv.most_similar(
    "excellent", topn=5
))

# STEP 8: Calculate Semantic Similarity

word_pairs = [
    ("excellent", "amazing"),
    ("poor", "terrible"),
    ("product", "quality")
]

print("\nSEMANTIC SIMILARITY USING CBOW")

for w1, w2 in word_pairs:

    score = cbow_model.wv.similarity(w1, w2)

    print(w1, "<->", w2, ":", round(float(score), 4))

print("\nSEMANTIC SIMILARITY USING SKIP-GRAM")

for w1, w2 in word_pairs:

    score = skipgram_model.wv.similarity(w1, w2)

    print(w1, "<->", w2, ":", round(float(score), 4))

# STEP 9: Save Both Models

cbow_model.save("cbow_model.model")

skipgram_model.save("skipgram_model.model")

print("\nBoth Word2Vec models saved successfully!")