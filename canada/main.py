import csv

def mySplit(arr):
    translated_text = arr.translate(str.maketrans({'\'': ' \' ', '.': ' . ','?': ' ? '}))
    return translated_text.split()

d = {}

with open('dinosaur_dataset.csv', 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        first = False
        for row in csv_reader:
            if not first:
                first = True
                continue
            
            for k,v in zip(mySplit(row[0]), mySplit(row[1])):
                 d[k] = v
# print(d)

sentences = ['sentence']
with open('test-input.csv', 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    first = False
    for row in csv_reader:
        f_word = ""
        sentence = []
        if not first:
            first = True
            continue
        for k in mySplit(row[0]):
            sentence.append(d[k]) 
        f_word = " ".join(sentence[:-1])
        # f_word = " ".join(sentence)
        f_word = "".join([f_word, sentence[-1]])
        sentences.append(f_word)
        # print(f_word)
fileContent = "\n".join(sentences)
print(fileContent)
with open("output.csv", "w") as f:
     f.write(fileContent)