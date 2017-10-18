import numpy as np

class Stat:
    def __init__(self):
        self.vals = []

    def add(self, v):
        self.vals.append(v)

    def mean(self):
        return np.mean(self.vals, axis=0)

    def std(self):
        return np.std(self.vals, axis=0)

