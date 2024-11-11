import numpy as np
from abc import ABC, abstractmethod


class ObjectiveFunction(ABC):

    def __init__(self, x: np.ndarray):
        self.x = x

    @abstractmethod
    def compute_value_at_x(self):
        pass

    @abstractmethod
    def compute_gradient_at_x(self):
        pass


class Rosenbrock(ObjectiveFunction):
    """The two-dimensional Rosenbrock function is also known as the banana function. It was introduced by Rosenbrock, who used it as a benchmark problem for optimize algorithms.
    Args:
        x (np.ndarray): Input vector for the function
    """

    # Implement the computation of the objective function value at point x here. Do not change anything above this line.
    def compute_value_at_x(self) -> float:
        return float((1 - self.x[0]) ** 2 + 100 * (self.x[1] - self.x[0] ** 2) ** 2)

    def compute_gradient_at_x(self) -> np.ndarray:
        return np.array([2 * (self.x[0] * (1 - 200 * (self.x[1] - self.x[0] ** 2)) - 1), 200 * (self.x[1] - self.x[0] ** 2)])
