from collections import Counter
import numpy as np

class SPRINT():
    def __init__(self, x_train, y_train):
        self.dataset = x_train
        self.labels = y_train
        self.join_dataset_label()

    def join_dataset_label(self):
      x = np.zeros((self.dataset.shape[0], self.dataset.shape[1]+1))
      for i, elem in enumerate(self.dataset):
        x[i][:self.dataset.shape[1]]=elem[:self.dataset.shape[1]]
        x[i][-1] = self.labels[i]
      self.dataset = x

    def calculate_gini_index(self, groups, classes):
        # Calculate the Gini index for a split dataset
        n_instances = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            class_counts = Counter([row[-1] for row in group])
            for class_val in classes:
                p = class_counts[class_val] / size
                score += p * p
            gini += (1.0 - score) * (size / n_instances)
        return gini

    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.calculate_gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def split(self, node, max_depth, min_size, depth):
        left, right = node['groups']
        del(node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], max_depth, min_size, depth + 1)
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], max_depth, min_size, depth + 1)

    def fit(self, max_depth, min_size):
        root = self.get_split(self.dataset)
        self.split(root, max_depth, min_size, 1)
        self.root = root

    def predict(self, test_data):
        preds = np.zeros(len(test_data), dtype='int32')
        for i, row in enumerate(test_data):
            pred = self.test(self.root, row)
            preds[i] = pred
        return preds

    def test(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.test(node['left'], row)
            else:
                return (node['left'])
        else:
            if isinstance(node['right'], dict):
                return self.test(node['right'], row)
            else:
                return (node['right'])
