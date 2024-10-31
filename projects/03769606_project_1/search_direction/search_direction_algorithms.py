from abc import ABC, abstractmethod
import numpy as np


class SearchDirectionAlgorithm(ABC):

    @staticmethod
    @abstractmethod
    def compute_search_direction(objective_function):
        pass


class SteepestDescent(SearchDirectionAlgorithm):

    # Implement the computation of the normalized search direction here. Do not change anything above this line.
    @staticmethod
    def compute_search_direction(gradient: np.ndarray) -> np.ndarray:
        return - gradient / np.linalg.norm(gradient)
