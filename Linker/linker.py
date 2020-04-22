from itertools import chain
import csv

import pandas


def print_csv(csv_path):
    """
    This function takes as argument the path of a csv file and then it prints the csv file.
    """
    with open(csv_path) as csv_path:
        reader = csv.reader(csv_path)
        for row in reader:
            print("\t | ".join(row))


def print_csv_column(csv_path, column_number):
    """
    This function takes as arguments the path of a csv file and the number of a specific column and then it prints the column.
    """
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            print(f'\t{row[column_number]}')


def csv_to_list(csv_path):
    """
        This function takes as argument the path of a csv file and then it transforms it to a list.
        """
    with open(csv_path, newline='') as csv_file:
        reader = csv.reader(csv_file)
        data = list(reader)
        return data


def csv_to_1d_list(csv_path):
    """
    This function takes as argument the path of a csv file and then it transforms it to an 1-D list.
    """
    with open(csv_path, newline='') as csv_file:
        reader = csv.reader(csv_file)
        data = list(reader)
        data = list(chain.from_iterable(data))
        return data


def csv_column_to_list(csv_path, column_number):
    """
    This function takes as argument the path of a csv file and the number of a specific column and then it transforms it to a list.
    """
    my_list = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            my_list.append(row[column_number])
            # print(f'\t{row[column_number]}')
    return my_list


def csv_spliter(csv_path, percentage, csv_first_path, csv_second_path):
    all_data = csv_to_list(csv_path)
    all_data_length = all_data.__len__()
    all_data_starting_length_to_test = all_data_length * percentage / 100
    all_data_to_test = []
    for i in range(0, int(all_data_starting_length_to_test)):
        all_data_to_test.append(all_data[i])

    pandas.DataFrame(all_data_to_test).to_csv(csv_first_path, header=None, index=None)

    all_data_to_be_tested = []
    for i in range(int(all_data_starting_length_to_test), all_data.__len__()):
        all_data_to_be_tested.append(all_data[i])
    pandas.DataFrame(all_data_to_be_tested).to_csv(csv_second_path, header=None, index=None)


def csv_clear(csv_path):
    f = open(csv_path, "w+")
    f.close()
