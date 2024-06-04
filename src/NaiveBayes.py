from collections import Counter
import numpy as np


class NaiveBayes:
    def __init__(self, min_categories):
        self.min_categories = min_categories
        self.priors = {}
        self.likelihoods = {}
        

    def fit(self, train_features, train_classes):
        self.priors = Counter(train_classes)
        self.likelihoods = np.zeros(
            shape=(len(self.priors), train_features.shape[1]), dtype=object
        )
        for row in self.likelihoods:
            for i, _ in enumerate(row):
                row[i] = np.zeros(self.min_categories[i])
            
        for features, result_class in zip(train_features, train_classes):
            for i, feature in enumerate(features):
                self.likelihoods[result_class, i][feature] += 1
        # total = self.priors.total()
        total = sum(self.priors.values())
        for key in self.priors.keys():
            key_occurances = self.priors[key]
            self.priors[key] /= total
            for i in range(train_features.shape[1]):  # Laplace smoothing
                for j in range(self.min_categories[i]):
                    self.likelihoods[key, i][j] = (self.likelihoods[key, i][j] + 1) / (
                        key_occurances + self.min_categories[i]
                    )

    def predict(self, X):
        predictions = np.zeros(len(X), dtype='int32')
        for i, sample in enumerate(X):
            max_probability = float("-inf")
            for key in self.priors.keys():
                probability = np.log(self.priors[key])
                for i, feature in enumerate(sample):
                    probability += np.log(self.likelihoods[key, i][feature])
                if probability > max_probability:
                    prediction = key
                    max_probability = probability
            predictions[i] = prediction
        return predictions
