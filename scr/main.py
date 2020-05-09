import pandas
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import time
import tensorflow
from keras.callbacks import EarlyStopping

start_time = time.time()

ps = PorterStemmer()
tf_idf = TfidfVectorizer()
stop_words = set(stopwords.words('english'))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

file = pandas.read_csv("../Input/onion-or-not.csv")
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
# print(tf_idf.get_feature_names())

x = tf_idf.transform(vector_text)
# print(x.shape)
# print(x)
# print(x.toarray())
df = pandas.DataFrame(x.toarray(), columns=tf_idf.get_feature_names())
print("\n\n\n")
# pandas.set_option('display.max_columns', None)
# print(len(df))
# print(len(label))
df.insert(len(df.columns), "labelz", label, True)
# pandas.set_option('display.max_rows', None)
print(df)
print(df.size)

X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'labelz'], df.labelz, test_size=0.25)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20)

print(X_train)

# X_train = tensorflow.keras.utils.normalize(X_train, axis=1)
# X_test = tensorflow.keras.utils.normalize(X_test, axis=1)
# y_train = tensorflow.keras.utils.normalize(y_train, axis=1)

model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Flatten())
model.add(tensorflow.keras.layers.Dense(128, activation=tensorflow.nn.relu))
model.add(tensorflow.keras.layers.Dense(128, activation=tensorflow.nn.relu))
model.add(tensorflow.keras.layers.Dense(2, activation=tensorflow.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train.to_numpy(), y_train.to_numpy(), epochs=10, batch_size=16, verbose=1, validation_data=(X_test.to_numpy(), y_test.to_numpy()), callbacks=[es])

val_loss, val_acc = model.evaluate(X_test, y_test)
print("\n\nLoss: \t\t", val_loss, "\nAccuracy: \t", val_acc, "\n\n")

# model.save("save.model")
# new_model = tensorflow.keras.models.load_model("save.model")
predictions = model.predict(X_test.to_numpy())
# print(predictions)

predictions_list = list()
for k in range(len(predictions)):
    predictions_list.append(numpy.argmax(predictions[k]))

# print(numpy.argmax(predictions[0]))
# print(numpy.argmax(predictions[3]))

print("F1 Score: \t\t\t", round(f1_score(y_test, predictions_list, average='micro'), 4))
print("Precision Score: \t", round(precision_score(y_test, predictions_list, average='micro'), 4))
print("Recall Score: \t\t", round(recall_score(y_test, predictions_list, average='micro'), 4), "\n\n")

print("--- %s seconds ---" % (time.time() - start_time))
