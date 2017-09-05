# Numerical utilities
import numpy as np


class Nonlinearity:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + (np.exp(-1 * x)))


