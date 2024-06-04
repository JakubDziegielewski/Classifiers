from sklearn.preprocessing import LabelEncoder
from bisect import bisect_right
from sklearn.model_selection import train_test_split
from scipy.io import arff
import pandas as pd
import numpy as np


def prepare_data(file_name, col_to_drop, group_vector):
    file = arff.loadarff("data/{}".format(file_name))
    df = pd.DataFrame(file[0])
    if col_to_drop != None:
        df = df.drop(col_to_drop, axis=1)
    df["class"] = encode_column(df["class"])
    X = np.array(df.drop("class", axis=1))
    y = np.array(df["class"])
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=3, stratify=y
    )
    intervals = find_intervals(x_train, group_vector)
    x_train = np.array(
        [
            data_discretization(features, intervals[i])
            for i, features in enumerate(x_train.T)
        ]
    ).T
    x_test = np.array(
        [
            data_discretization(features, intervals[i])
            for i, features in enumerate(x_test.T)
        ]
    ).T

    return x_train, x_test, y_train, y_test


def encode_column(column):
    le = LabelEncoder()
    column = le.fit_transform(column)
    return column


def find_intervals(x_train, group_vector):
    intervals = [np.zeros(i - 1) for i in group_vector]

    for i, features in enumerate(x_train.T):
        max_value = max(features)
        min_value = min(features)
        section_size = (max_value - min_value) / group_vector[i]
        intervals[i] = np.array(
            [min_value + section_size * j for j in range(1, group_vector[i])]
        )
    return intervals


def data_discretization(data, intervals):
    return [bisect_right(intervals, x) for x in data]
