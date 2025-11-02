
### How It Works

The script:
1. Reads and tokenizes `corpus.txt` using NLTK
2. Trains Word2Vec embeddings with 100K epochs
3. Solves analogies using vector arithmetic: `A - B + C = D`
4. Filters results to profession words only
5. Generates and saves the flag to `flag.txt`

### Analogy Formula

For analogy `A : B :: C : D` (A is to B as C is to D):
- Compute: `D ≈ C + (A - B)`
- In code: `positive=[A, C], negative=[B]`

## 📂 Files

- `corpus.txt` - Training corpus about professions
- `main.py` - Solution script
- `flag.txt` - Generated flag (output)

## 🚩 Expected Output

```
SIGMOID_ENGINEERS_ATHLETES
```
