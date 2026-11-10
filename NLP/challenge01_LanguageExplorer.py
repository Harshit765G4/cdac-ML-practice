sentence = input("Enter a sentence: ")

print("\nInput :", sentence)

print("\nTotal Characters: ", len(sentence))

words = sentence.split()
print("\nIndividual Words: ", words)

print("\nTotal Words: ", len(words))

print("\nFirst Word: ", words[0] if words else "No words found")
print("\nLast Word: ", words[-1] if words else "No words found")

count = 0
for word in words:
    if word.lower().strip(".,!?;:") == "python":
        count += 1

print(f"\nThe word 'python' appears {count} times in the sentence.")

print()

vowelCount = 0
vowels = "aeiou"
for i in range(len(sentence)):
    if sentence[i].lower() in vowels:
        print(f"Vowel '{sentence[i]}' found at index {i}")
        vowelCount += 1

print(f"\nTotal vowels found: {vowelCount}")