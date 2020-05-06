import pandas as pd
import numpy as np
import nltk

#nltk.download('stopwords')
#nltk.download('punkt')
df = pd.read_csv("..\\Input\\onion-or-not.csv", index_col=False)
dataset=df.filter(['text', 'label'], axis=1)

df.loc[:, "text"] = df.text.apply(lambda x: str.lower(x))
import re
df.loc[:, "text"] = df.text.apply(lambda x : " ".join(re.findall('[\w]+', x)))

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def remove_stopWords(s):
    """
    For removing stop words
    """
    s = ' '.join(word for word in s.split() if word not in stop_words)
    return s


df.loc[:, "text"] = df.text.apply(lambda x: remove_stopWords(x))

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
df['text'] = df['text'].str.split()
df['stemmed'] = df['text'].apply(lambda x: [stemmer.stem(y) for y in x])    # Stem every word.
df = df.drop(columns=['text'])
print(df)
