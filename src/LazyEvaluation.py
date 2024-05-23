import numpy as np
from collections import Counter

class ContrastPatternClassificator:
    def __init__(self, X: np.array, y:np.array) -> None:
        self.X = X
        self.y = y
        self.classes = np.unique(y)
    def find_contrast_patterns(self, x: np.array):
        contrast_patterns = dict(zip(self.classes, [] * len(self.classes)))
        possible_patterns = x == self.X
        for pattern, target_class in zip(possible_patterns, self.y):
            indices = np.where(pattern)
            size = np.size(indices)
            if size == 0:
                continue
            current_shortest_contrast = contrast_patterns[target_class]
            if current_shortest_contrast is not None and np.size(current_shortest_contrast) <= size:
                continue
            #iterate over data, stop if two classes match the pattern
                
        return contrast_patterns