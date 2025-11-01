with open('CUBA/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Remove periods at the end of sentences
text = text.replace('.', '')

# Write to output.txt
with open('CUBA/output.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print("Processed text saved to output.txt")