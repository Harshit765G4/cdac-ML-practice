sentence = "AI is amazing. AI is changing the world, and NLP is amazing!"

print("\nTotal Characters:", len(sentence))

words = sentence.split()
print("\nWord Count:", len(words))

unique_words = set()

for word in words:
    clean_word = word.lower().strip(",.!?:;")
    
    unique_words.add(clean_word)
    
print("\nTotal Unique Words:", len(unique_words))
print("Unique Words:", unique_words)

freq = {}
for word in words:
    normm_word = word.lower().strip(",.!?;:")
    freq[normm_word] = freq.get(normm_word, 0) + 1
    
print("\nFrequency of every individual word:", freq)

most_freq_word = max(freq, key=freq.get)

print("\nMost frequently occurring word:", most_freq_word)

if words:
    first_word = words[0]
    last_word = words[-1]

    print(f"\nFirst Word: {first_word}, Last Word: {last_word}")
else:
    print("Please enter a valid sentence.")