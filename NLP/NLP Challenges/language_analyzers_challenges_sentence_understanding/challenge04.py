sentence = """
NLP is powerful. NLP helps computers understand language.
Language is complex, but NLP makes language processing easier.
"""

words = sentence.lower().split()
print("\nWords Extracted:", words)

cleanWords = {}

for word in words:
    clean_word = word.strip(".,!?;:")
    cleanWords[clean_word] = cleanWords.get(clean_word, 0) + 1

print("\nWords with Individual Frequency:", cleanWords)
print("\nTotal Number of Tokens:", len(words))

print("\nUnique Number of Tokens:", len(cleanWords))

top_3 = sorted(cleanWords.items(), key=lambda x: x[1], reverse=True)[:3]
print("\nTop 3 Frequent Words:", top_3)

vocabulary_diversity = len(cleanWords) / len(words)
print("\nVocabulary Diversity:",round(vocabulary_diversity, 5))

print("\nVocabulary Diversity Percentage:",
      round(vocabulary_diversity, 5) * 100, "%")