from shlex import join


sentence = "Natural Language Processing is Interesting"

print("Original Sentence: ")
print(sentence)

print("\nTotal characters: ")
print(len(sentence))

words = sentence.split()

print("\nIndividual Words: ")
print(words)

print("\nTotal Words: ")
print(len(words))

print("\nWord-by-word Analysis: ")

for index, word in enumerate(words, start = 1):
    print(f"Word {index}: {word}")


usrSentence = input("Enter a Sentence: ")

print("\nCount of Characters: ")
print(len(usrSentence))

usrWords = usrSentence.split()

print("\nIndividual Words Displaying: ")
print(usrWords)

print("\nTotal Number of Words: ")
print(len(usrWords))

first_word = usrWords[0]
last_word = usrWords[-1]

print(f"First Word: {first_word}, Last Word: {last_word}")