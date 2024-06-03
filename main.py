from src.NaiveBayes import NaiveBayes
from src.LazyEvaluation import ContrastPatternClassificator
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from bisect import bisect_right


nb = NaiveBayes()
file = arff.loadarff("data/glass.arff")
df = pd.DataFrame(file[0])


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


df["class"] = encode_column(df["class"])
X = np.array(df.drop("class", axis=1))
#X = np.array(df.drop("class", axis=1).drop("IDNumber", axis=1))
y = np.array(df["class"])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=3)
intervals = find_intervals(x_train, [10, 10, 10, 10, 10, 10, 8, 10, 8, 10])
x_train = np.array([data_discretization(features, intervals[i]) for i, features in enumerate(x_train.T)]).T
x_test = np.array([data_discretization(features, intervals[i]) for i, features in enumerate(x_test.T)]).T
lazy_clf = ContrastPatternClassificator(x_train, y_train)

pred = lazy_clf.predict(x_test)
print(f"Lazy classification accuracy: {sum(pred == y_test)/len(y_test)}")

nb = NaiveBayes()
nb.fit(x_train, y_train)
pred = nb.predict(x_test)
print(f"Naive bayes accuracy: {sum(pred == y_test) / len(y_test)}")

"""
x_train = np.loadtxt("data/X_train.txt", dtype = float)
x_test = np.loadtxt("data/X_test.txt", dtype = float)
y_train = np.loadtxt("data/y_train.txt", dtype = int)
y_test = np.loadtxt("data/y_test.txt", dtype = int)
verbose = True
nb = NaiveBayes()
intervals = find_intervals(x_train, [4] * 561)
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
if verbose:
    print(f"Testing discrete naive bayes classifier")
    print("result of classification of the test set:")
good = 0
total = 0
nb = NaiveBayes()
nb.fit(x_train, y_train - 1)
predictions = nb.predict(x_test)

if verbose:
    print(f"Accuracy: {sum(predictions == y_test - 1) / len(predictions)}")
"""