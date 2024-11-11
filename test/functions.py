"""
Contains functions for test optimize algorithms.
Source: https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

import numpy as np


def rosenbrock(x: np.ndarray, a: float = 1, b: float = 100) -> float:
    f: float = 0
    for i in range(len(x) - 1):
        f += (a - x[i]) ** 2 + b * (x[i + 1] - x[i] ** 2) ** 2
    return f
