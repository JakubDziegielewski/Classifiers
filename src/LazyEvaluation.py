import numpy as np
from itertools import repeat
from collections import Counter


class ContrastPatternClassificator:
    def __init__(self, X: np.array, y: np.array) -> None:
        self.X = X
        self.y = y
        self.class_counter = Counter(y)
        self.classes = self.class_counter.keys()
        self.reduced_table = None

    def _find_reduced_table_patterns(self, x: np.array):
        possible_patterns = x == self.X
        sorted_table = sorted(
            np.column_stack([possible_patterns, self.y]), key=lambda x: sum(x[:-1])
        )
        self.reduced_table = sorted_table

    def _find_possible_patterns(self, x: np.array):
        possible_patterns = dict(zip(self.classes, repeat(0.0)))
        found_patterns = 0
        for pattern in self.reduced_table:
            if sum(pattern[:-1]) == 0:
                continue
            cl = pattern[-1]
            indices = np.where(pattern[:-1])
            matching_cases = 0
            for matched_pattern in self.reduced_table:
                if not np.equal(matched_pattern[indices], pattern[indices]).all():
                    continue
                if cl != matched_pattern[-1]:
                    break
                matching_cases += 1
            else:
                possible_patterns[cl] = matching_cases / self.class_counter[cl]
                found_patterns += 1
            if found_patterns == len(self.classes):
                break
        return possible_patterns
            
    def predict(self, x:np.array):
        self._find_reduced_table_patterns(x)
        possible_patterns = self._find_possible_patterns(x)
        max_prob = 0
        cl = None
        for k, v in possible_patterns.items():
            if v == 0.0:
                continue
            if v > max_prob:
                cl = k
                max_prob = v
        if max_prob == 0:
            return self.class_counter.most_common(1)[0][0]
        return cl

