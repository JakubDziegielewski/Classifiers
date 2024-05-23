from  src.NaiveBayes import NaiveBayes
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder


def run_naive_bayes(test_size, random_state, x, y, verbose=False, group_number=4):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    nb = NaiveBayes()
    nb.find_intervals(x_train, group_number)
    intervals = nb.intervals
    x_train = np.array(
        [
            NaiveBayes.data_discretization(features, intervals[i])
            for i, features in enumerate(x_train.T)
        ]
    ).T
    x_test = np.array(
        [
            NaiveBayes.data_discretization(features, intervals[i])
            for i, features in enumerate(x_test.T)
        ]
    ).T

    if verbose:
        print(f"Testing discrete naive bayes classifier")
        print(f"test size = {test_size}; random_state = {random_state}")
        print("result of classification of the test set:")
    good = 0
    total = 0
    nb = NaiveBayes()
    nb.build_classifier(x_train, y_train, group_number)
    for test_x, test_y in zip(x_test, y_test):
        prediction = nb.predict(test_x)
        if verbose:
            print(f"Prediction: {prediction}, True class: {test_y}")
        if prediction == test_y:
            good += 1
        total += 1
    if verbose:
        print(f"Accuracy: {good/total:.3f}")
    return good / total


file = arff.loadarff("data/wdbc.arff")
df = pd.DataFrame(file[0])


def encode_column(column):
    le = LabelEncoder()
    column = le.fit_transform(column)
    return column

df["class"] = encode_column(df["class"])
data = np.array(df.drop("class", axis=1))

run_naive_bayes(0.1, 1, data, df["class"], group_number=4, verbose=True)