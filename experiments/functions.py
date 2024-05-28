from sklearn.preprocessing import LabelEncoder
from bisect import bisect_right
import numpy as np

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