import linker


import csv
#with open("C:\\Users\\alexd\\PycharmProjects\\fakenews\\onion-or-not.csv", "r", encoding="utf8") as source:
    #reader = csv.reader(source)
#with open("C:\\Users\\alexd\\PycharmProjects\\fakenews\\onion-or-not.csv", "w", newline='', encoding="utf8") as result:


#dataset = "C:\\Users\\alexd\\PycharmProjects\\fakenews\\onion-or-not.csv"
    #training_set = "C:\\Users\\alexd\\PycharmProjects\\fakenews\\onion training set.csv"
    #testing_set = "C:\\Users\\alexd\\PycharmProjects\\fakenews\\onion testing set.csv"
    #linker.csv_clear(training_set)
    #linker.csv_clear(testing_set)

data = linker.csv_to_list("C:\\Users\\alexd\\PycharmProjects\\fakenews\\onion-or-not.csv")
data.pop(0)
#data = [int(i) for i in data]

print(data)

