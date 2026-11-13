import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

# Added encoding='latin1' to handle special non-UTF-8 characters in the CSV file
df = pd.read_csv('DisneylandReviews.csv', encoding='latin1')

review = str(df['Review_Text'][0])

print("Original Review:")
print(review)

print("\nSentence Tokenization")
sentences = sent_tokenize(review)
for s in sentences:
    print(s)

print("\nWord Tokenization")
words = word_tokenize(review)
print(words)

print("\nStatistics")
print("Number of Sentences:", len(sentences))
print("Number of Words:", len(words))

