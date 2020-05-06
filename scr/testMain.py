import re
import pandas
from Linker import functions
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

file = pandas.read_csv("../Input/onion-or-not.csv")
text = file.text
label = file.label

vector_text = text.to_numpy()
print(vector_text)
vectors_of_words = []
vectors_of_strings = []
for strings in range(len(vector_text)):
    # print(vector_text[strings])
    for word in word_tokenize(vector_text[strings]):
        # print(word)
        if ps.stem(word) not in stop_words:
            word = ps.stem(word)
            print(word)
        else:
            vector_text[strings].replace(word, "reeee")
            print(vector_text[strings])

print("\n")
# print(vector_text)

vector_of_words = functions.stemming(vectors_of_words)

# vector_of_lower_words = [x.lower() for x in vector_of_words]
# print(vector_of_words)

print("\n")

print("Length before stopwords: ", len(vector_of_words))
vector_of_right_words = functions.stop_words(vector_of_words)
print("Length after stopwords: ", len(vector_of_right_words))
print("\n")

# print(list(string.ascii_lowercase))

# for word in range(len(list_of_right_words)):
#     print("Will be used to eliminate useless characters!")



