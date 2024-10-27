from typing import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class Interval:
    bounds: tuple[float, float] = (-np.inf, np.inf)
    closed: tuple[bool, bool] = (True, True)

    def __post_init__(self):
        if self.bounds[0] > self.bounds[1]:
            raise ValueError('Lower bound cannot be larger than upper bound.')

    def __str__(self):
        return '[' if self.closed[0] else '(' + f'{self.bounds[0]}, {self.bounds[1]}' + ']' if self.closed[1] else ')'


def gradient_in_bounds(f: Callable[[np.ndarray], float], bounds: list[Interval], eps: float = 1e-8) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a function computing the gradient of a function `f` as a function of `x` without exceeding the specified bounds.
    Useful if `f` is not computable outside these bounds, due to negative arguments being passed to numpy.sqrt, for example.
    :param f: function to compute the gradient of
    :param bounds: bounds as a list of Interval objects
    :param eps: gradient step size
    :return: gradient of `f` as a function of `x`
    """

    def gradient(x):
        if len(bounds) != len(x):
            raise ValueError('The number of bounds must equal the number of arguments.')

        g = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] + eps <= bounds[i].bounds[1]:
                dx: float = eps
            elif x[i] - eps >= bounds[i].bounds[0]:
                dx: float = -eps
            else:
                raise ValueError(f'Gradient cannot be computed because x{i} +- {eps} is out of bounds {bounds[i]}. Consider loosening the bounds or decreasing the step size.')
            g[i] = (f(x + dx) - f(x)) / dx
        return g

    return gradient
