from src.NaiveBayes import NaiveBayes
from src.LazyEvaluation import ContrastPatternClassificator
from src.sprint import SPRINT
from experiments.functions import prepare_data
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import precision_score, recall_score, make_scorer, accuracy_score
from sklearn.model_selection import StratifiedKFold


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

def cross_validation(classifier_name, X, y, cv, min_categories = None):
    kf = StratifiedKFold(n_splits=cv)
    acc_scores = []

    for train_idx, test_idx in kf.split(X, y):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if classifier_name == "sprint":
            obj = SPRINT(x_train, y_train)
            obj.fit(4, 5)
            preds = obj.predict(x_test)
        if classifier_name == "bayes":
            nb = NaiveBayes(min_categories)
            nb.fit(x_train, y_train)
            preds = nb.predict(x_test)
        if classifier_name == "lazy":
            lazy_clf = ContrastPatternClassificator(x_train, y_train)
            preds = lazy_clf.predict(x_test)

        lazy_acc = sum(preds == y_test) / len(y_test)
        acc_scores.append(lazy_acc)

    avg_accuracy = np.mean(acc_scores)
    return acc_scores


def run_experiments(file_group_list: list, col_to_drop: List[str] = None, save_path="../imgs/"):
    dict_list = []
    for i, (file_name, group_vector) in enumerate(file_group_list):
        dict_acc = {}
        dict_prec = {}
        dict_rec = {}
        dict_time = {}

        x_train, x_test, y_train, y_test = prepare_data(file_name, col_to_drop[i], group_vector)

        # lazy
        runtime = 0
        for _ in range(10):
            start_time = time.time()
            lazy_clf = ContrastPatternClassificator(x_train, y_train)
            pred = lazy_clf.predict(x_test)
            end_time = time.time()
            runtime += (end_time - start_time)
        runtime /= 10.0
        lazy_acc = sum(pred == y_test) / len(y_test)
        print(cross_validation("lazy", x_train.copy(), y_train.copy(), 4))
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
        dict_time["lazy"] = runtime

        # bayes
        runtime = 0
        for _ in range(10):
            start_time = time.time()
            nb = NaiveBayes(group_vector)
            nb.fit(x_train, y_train)
            pred = nb.predict(x_test)
            end_time = time.time()
            runtime += (end_time - start_time)
        runtime /= 10.0
        nb = NaiveBayes(group_vector)
        bayes_acc = sum(pred == y_test) / len(y_test)
        print(cross_validation("sprint", x_train.copy(), y_train.copy(), 4))
        print(cross_validation("bayes", x_train.copy(), y_train.copy(), 4, min_categories=group_vector))

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
        dict_time["bayes"] = runtime

        # sprint
        max_depth = 4
        min_size = 5
        runtime = 0
        for _ in range(10):
            start_time = time.time()
            obj = SPRINT(x_train, y_train)
            obj.fit(max_depth, min_size)
            preds = obj.predict(x_test)
            end_time = time.time()
            runtime += (end_time - start_time)
        runtime /= 10.0
        sprint_acc = sum((preds==y_test))/len(y_test)
        print(cross_validation("sprint", x_train.copy(), y_train.copy(), 4))

        if len(set(y_test)) > 2:
            sprint_prec = precision_score(y_test, pred, average='macro')
            sprint_rec = recall_score(y_test, pred, average='macro')
        else:
            sprint_prec = precision_score(y_test, pred, pos_label=0)
            sprint_rec = recall_score(y_test, pred, pos_label=0)
        print(f"Sprint accuracy: {sprint_acc} %")
        dict_acc["sprint"] = sprint_acc
        dict_prec["sprint"] = sprint_prec
        dict_rec["sprint"] = sprint_rec
        dict_time["sprint"] = runtime

        dict_list.append(dict_acc)
        dict_list.append(dict_prec)
        dict_list.append(dict_rec)
        dict_list.append(dict_time)
        ylabel = ["Accuracy [%]", "Precision [%]", "Recall [%]", "Time [s]"]

    for i in range(4):
        dicts = [dict_list[i], dict_list[i + 4]]
        plot(dicts, file_group_list, ylabel[i])
