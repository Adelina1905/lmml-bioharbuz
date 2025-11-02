import numpy as np
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
print("📦 Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Read and preprocess corpus
print("\n📖 Reading corpus...")
with open('MEXICO/corpus.txt', 'r', encoding='utf-8') as f:
    corpus_text = f.read()

print("🔤 Advanced tokenizing with NLTK...")
raw_sentences = sent_tokenize(corpus_text)

sentences = []
for sent in raw_sentences:
    tokens = word_tokenize(sent.lower())
    tokens = [word for word in tokens if word.isalpha() and len(word) > 1]
    if len(tokens) > 2:
        sentences.append(tokens)

print(f"   Found {len(sentences)} sentences")
print(f"   Total tokens: {sum(len(s) for s in sentences)}")

# ULTRA EXTREME TRAINING
print("\n🧠 Training Word2Vec model with ULTRA EXTREME parameters...")
model = Word2Vec(
    sentences=sentences,
    vector_size=200,       # Larger vectors
    window=20,             # VERY large window
    min_count=1,          
    workers=4,
    epochs=1000,         # 🔥 200K EPOCHS!!!
    sg=1,                  # Skip-gram
    negative=20,           # More negative sampling
    alpha=0.1,             # Very high learning rate
    min_alpha=0.00001,     # Keep learning
    hs=0,                  # Negative sampling only
    sample=0,              # No downsampling
    seed=42
)

print(f"✓ Model trained! Vocabulary size: {len(model.wv)}")

# Check learned relationships
print("\n📊 Checking learned relationships:")
relationships = [
    ('doctors', 'medicine'),
    ('engineers', 'law'),
    ('teachers', 'schools'),
    ('athletes', 'hospitals')
]

for word1, word2 in relationships:
    if word1 in model.wv and word2 in model.wv:
        sim = model.wv.similarity(word1, word2)
        print(f"   {word1} <-> {word2}: {sim:.4f}")

# Define professions to filter
PROFESSIONS = ['doctors', 'engineers', 'teachers', 'athletes', 'nurses', 'lawyers']

print("\n🔍 Solving analogies using Gensim's built-in method...")

# Analogy 1: doctors:medicine::?:law
# This means: ? = law - medicine + doctors
print("\n1️⃣ doctors - medicine + law = ?")
print("   Computing: law - medicine + doctors")
try:
    results1 = model.wv.most_similar(
        positive=['law', 'doctors'],
        negative=['medicine'],
        topn=20
    )
    print("\n   All top 20 results:")
    for word, sim in results1:
        marker = "👤" if word in PROFESSIONS else "  "
        print(f"      {marker} {word}: {sim:.4f}")
    
    # Find first profession
    word1 = None
    for word, sim in results1:
        if word in PROFESSIONS and word != 'doctors':
            word1 = word
            print(f"\n   → Selected (first profession): {word1}")
            break
    
    if not word1:
        word1 = results1[0][0]
        print(f"\n   → Selected (top result): {word1}")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    word1 = "UNKNOWN"

# Analogy 2: teachers:schools::?:hospitals
# This means: ? = hospitals - schools + teachers
print("\n2️⃣ teachers - schools + hospitals = ?")
print("   Computing: hospitals - schools + teachers")
try:
    results2 = model.wv.most_similar(
        positive=['hospitals', 'teachers'],
        negative=['schools'],
        topn=20
    )
    print("\n   All top 20 results:")
    for word, sim in results2:
        marker = "👤" if word in PROFESSIONS else "  "
        print(f"      {marker} {word}: {sim:.4f}")
    
    # Find first profession
    word2 = None
    for word, sim in results2:
        if word in PROFESSIONS and word != 'teachers':
            word2 = word
            print(f"\n   → Selected (first profession): {word2}")
            break
    
    if not word2:
        word2 = results2[0][0]
        print(f"\n   → Selected (top result): {word2}")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    word2 = "UNKNOWN"

# Generate flag
if word1 and word2:
    flag = f"SIGMOID_{word1.upper()}_{word2.upper()}"
    print("\n" + "=" * 60)
    print("🚩 FLAG:")
    print(f"   {flag}")
    print("=" * 60)
    
    with open('MEXICO/flag.txt', 'w') as f:
        f.write(flag)
    print("\n💾 Saved to MEXICO/flag.txt")