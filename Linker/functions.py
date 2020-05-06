from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def stemming(list_name):
    vector_of_words = []
    for vectors in range(len(list_name)):
        # print(vector_of_words[lists_of_words])
        for word in range(len(list_name[vectors])):
            # print(vector_of_words[lists_of_words][word])
            # print(ps.stem(vector_of_words[lists_of_words][word]))
            vector_of_words.append(ps.stem(list_name[vectors][word]))
    return vector_of_words


def stop_words(list_name):
    vector_of_right_words = []
    stop_words = set(stopwords.words('english'))
    for word in range(len(list_name)):
        if list_name[word] not in stop_words:
            vector_of_right_words.append(list_name[word])
    return vector_of_right_words
