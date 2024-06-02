from collections import Counter
import numpy as np


class NaiveBayes:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}

    def fit(self, train_features, train_classes):
        group_vector = [max(x) + 1 for x in train_features.T]
        self.priors = Counter(train_classes)
        self.likelihoods = np.zeros(
            shape=(len(self.priors), train_features.shape[1]), dtype=object
        )
        for row in self.likelihoods:
            for i, _ in enumerate(row):
                row[i] = np.zeros(group_vector[i])
            
        for features, result_class in zip(train_features, train_classes):
            for i, feature in enumerate(features):
                self.likelihoods[result_class, i][feature] += 1
        # total = self.priors.total()
        total = sum(self.priors.values())
        for key in self.priors.keys():
            key_occurances = self.priors[key]
            self.priors[key] /= total
            for i in range(train_features.shape[1]):  # Laplace smoothing
                for j in range(group_vector[i]):
                    self.likelihoods[key, i][j] = (self.likelihoods[key, i][j] + 1) / (
                        key_occurances + group_vector[i]
                    )

    def predict(self, X):
        predictions = np.array([], dtype='int32')
        for sample in X:
            print(sample)
            max_probability = float("-inf")
            for key in self.priors.keys():
                probability = np.log(self.priors[key])
                for i, feature in enumerate(sample):
                    print(i)
                    probability += np.log(self.likelihoods[key, i][feature])
                if probability > max_probability:
                    prediction = key
                    max_probability = probability
            predictions = np.append(predictions, prediction)
        return predictions
