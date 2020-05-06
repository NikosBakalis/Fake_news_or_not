import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from Linker import functions
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import time

start_time = time.time()

ps = PorterStemmer()
tf_idf = TfidfVectorizer()
stop_words = set(stopwords.words('english'))

file = pandas.read_csv("../Input/onion-or-not-test.csv")
text = file.text
label = file.label

vector_text = text.to_numpy()
print(vector_text)
vectors_of_words = []
for strings in range(len(vector_text)):
    # print(vector_text[strings])
    for word in word_tokenize(vector_text[strings]):
        # print(word)
        vector_text[strings] = vector_text[strings].replace(word, ps.stem(word))
        if word in stop_words:
            vector_text[strings] = vector_text[strings].replace(word, "")
    # print(vector_text[strings])

vectors_of_strings = []

print("\n")
print(vector_text)

x = tf_idf.fit(vector_text)
print(x.vocabulary_)
print(tf_idf.get_feature_names())

x = tf_idf.transform(vector_text)
print(x.shape)
print(x)
print(x.toarray())
df = pandas.DataFrame(x.toarray(), columns=tf_idf.get_feature_names())
print("\n\n\n")
# pandas.set_option('display.max_columns', 2500)
# print(len(df))
# print(len(label))
df.insert(len(df.columns), "labelz", label, True)
# pandas.set_option('display.max_rows', None)
print(df)
# df.loc[:, df.columns != 'labelz']

X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'labelz'], df.labelz, test_size=0.25)

print("--- %s seconds ---" % (time.time() - start_time))
