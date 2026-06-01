from __future__ import annotations


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = float(mean)
        self.std = float(std)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
