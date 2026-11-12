sentence = "I love NLP, and I love Python!"

print("\nTotal Characters: ", len(sentence))

words = sentence.split()
print("\nWords Count: ", len(words))

loveCount = 0
# for word in words:
#     if word.lower().strip(".,!?;:") == "love":
#         loveCount += 1

iCount = 0
# for word in words:
#     if word.lower().strip(".,!?;:") == "i":
#         iCount += 1

uniqueWords = set()

for word in words:
    clean_word = word.lower().strip(".,!?;:")
    if clean_word == 'love':
        loveCount += 1
        
    if clean_word == 'i':
        iCount += 1
        
    uniqueWords.add(clean_word)
    
print(f"\nThe word 'love' appears {loveCount} times in the sentence.")

print(f"\nThe word 'I' appears {iCount} times in the sentence.")

print(f"\nUnique Words: {len(uniqueWords)}")
print("\nUnique Words:", uniqueWords)