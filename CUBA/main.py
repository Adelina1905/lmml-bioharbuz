import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
nltk.download('stopwords')


with open('CUBA/input.txt', 'r', encoding='utf-8') as f:
    sentences = f.readlines()

stop_words = set(stopwords.words('english'))
filtered_sentences = []

for sentence in sentences:
    if sentence:
        word_tokens = word_tokenize(sentence)
        filtered_words = [w for w in word_tokens if not w.lower() in stop_words]
        filtered_sentence = " ".join(filtered_words)
        #remove dots, commas, etc
        filtered_sentence = filtered_sentence.replace(" .", "").replace(" ,","").replace(" '", "")
        filtered_sentences.append(filtered_sentence)
        

with open('CUBA/output.txt', 'w', encoding='utf-8') as f:
    for sentence in filtered_sentences:
        f.write(sentence + '\n')

