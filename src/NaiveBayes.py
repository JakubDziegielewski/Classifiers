from collections import Counter
import numpy as np
import bisect

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class NaiveBayes:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}

    def build_classifier(self, train_features, train_classes, group_number):
        self.priors = Counter(train_classes)
        self.likelihoods = np.zeros(
            shape=(len(self.priors), train_features.shape[1], group_number)
        )
        for features, result_class in zip(train_features, train_classes):
            for i, feature in enumerate(features):
                self.likelihoods[result_class, i, feature] += 1
        total = self.priors.total()
        for key in self.priors.keys():
            key_occurances = self.priors[key]
            self.priors[key] /= total
            for i in range(train_features.shape[1]): #Laplace smoothing
                for j in range(group_number):
                    self.likelihoods[key, i, j] = (self.likelihoods[key, i, j] + 1) / (key_occurances + group_number)
                    
    def find_intervals(self, train_data, number_of_groups):
        self.intervals = np.zeros(shape=(train_data.shape[1], number_of_groups - 1))
        for i, features in enumerate(train_data.T):
            max_value = max(features)
            min_value = min(features)
            section_size = (max_value - min_value) / number_of_groups
            
            self.intervals[i] = np.array(
                [min_value + section_size * j for j in range(1, number_of_groups)]
            )
            
    @staticmethod
    def data_discretization(data, intervals):
        return [bisect.bisect_right(intervals, x) for x in data]

    def predict(self, sample):
        max_probability = float("-inf")
        prediction = None
        for key in self.priors.keys():
            probability = np.log(self.priors[key])
            for i, feature in enumerate(sample):
                probability += np.log(self.likelihoods[key, i, feature])
            if probability > max_probability:
                prediction = key
                max_probability = probability
        return prediction
