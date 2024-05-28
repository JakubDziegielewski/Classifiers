from src.NaiveBayes import NaiveBayes
from src.LazyEvaluation import ContrastPatternClassificator
from src.sprint import SPRINT
from experiments.functions import encode_column, find_intervals, data_discretization
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import arff
from typing import List, Tuple, Dict
from sklearn.metrics import precision_score, recall_score


def plot(dict_list: List[Dict], file_group_list, ylabel: str, save_path="./imgs/"):
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, "{}.png".format(ylabel))
    for i, series in enumerate(zip(dict_list, file_group_list)):
        series, name = series
        plt.scatter(list(series.keys()), list(series.values()), label=str(name[0].rsplit('.', 1)[0]))
    plt.xlabel('Classifier type')
    plt.ylabel(ylabel)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig(file_path)
    plt.show()

def run_experiments(file_group_list: Tuple[str, List], col_to_drop: List[str] = None, save_path="../imgs/"):
    dict_list = []
    for i, (file_name, group_vector) in enumerate(file_group_list):
        dict_acc = {}
        dict_prec = {}
        dict_rec = {}

        file = arff.loadarff("data/{}".format(file_name))
        df = pd.DataFrame(file[0])
        if col_to_drop[i] != None:
            df = df.drop(col_to_drop[i], axis=1)

        df["class"] = encode_column(df["class"])
        X = np.array(df.drop("class", axis=1))
        y = np.array(df["class"])
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=3)
        intervals = find_intervals(x_train, group_vector)
        x_train = np.array([data_discretization(features, intervals[i]) for i, features in enumerate(x_train.T)]).T
        x_test = np.array([data_discretization(features, intervals[i]) for i, features in enumerate(x_test.T)]).T

        # lazy
        lazy_clf = ContrastPatternClassificator(x_train, y_train)
        pred = lazy_clf.predict(x_test)
        lazy_acc = sum(pred == y_test) / len(y_test)
        if len(set(y_test)) > 2:
            lazy_prec = precision_score(y_test, pred, average='macro')
            lazy_rec = recall_score(y_test, pred, average='macro')
        else:
            lazy_prec = precision_score(y_test, pred, pos_label=0)
            lazy_rec = recall_score(y_test, pred, pos_label=0)

        print(f"Lazy classification accuracy: {lazy_acc}")
        dict_acc["lazy"] = lazy_acc
        dict_prec["lazy"] = lazy_prec
        dict_rec["lazy"] = lazy_rec

        # bayes
        nb = NaiveBayes()
        nb.fit(x_train, y_train)
        pred = nb.predict(x_test)
        bayes_acc = sum(pred == y_test) / len(y_test)
        if len(set(y_test)) > 2:
            bayes_prec = precision_score(y_test, pred, average='macro')
            bayes_rec = recall_score(y_test, pred, average='macro')
        else:
            bayes_prec = precision_score(y_test, pred, pos_label=0)
            bayes_rec = recall_score(y_test, pred, pos_label=0)
        print(f"Naive bayes accuracy: {bayes_acc}")
        dict_acc["bayes"] = bayes_acc
        dict_prec["bayes"] = bayes_prec
        dict_rec["bayes"] = bayes_rec

        # sprint
        max_depth = 4
        min_size = 5
        obj = SPRINT(x_train, y_train)
        obj.fit(max_depth, min_size)

        preds = obj.test(x_test)
        sprint_acc = sum((preds==y_test))/len(y_test)
        if len(set(y_test)) > 2:
            sprint_prec = precision_score(y_test, pred, average='macro')
            sprint_rec = recall_score(y_test, pred, average='macro')
        else:
            sprint_prec = precision_score(y_test, pred, pos_label=0)
            sprint_rec = recall_score(y_test, pred, pos_label=0)
        print(f"Accuracy is: {sprint_acc} %")
        dict_acc["sprint"] = sprint_acc
        dict_prec["sprint"] = sprint_prec
        dict_rec["sprint"] = sprint_rec
        dict_list.append(dict_acc)
        dict_list.append(dict_prec)
        dict_list.append(dict_rec)
        ylabel = ["Accuracy", "Precision", "Recall"]

    for i in range(3):
        dicts = [dict_list[i], dict_list[i + 3]]
        plot(dicts, file_group_list, ylabel[i])
