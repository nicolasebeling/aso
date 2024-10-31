from typing import Callable

import numpy as np


def create_grad_function(f: Callable[[np.ndarray], float], bounds: list[tuple], eps: float = 1e-8) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a function that numerically computes the gradient of the function `f` as a function of `x`. If bounds are specified, they are not exceeded.
    This is useful if `f` is cannot be computed outside these bounds, due to negative arguments being passed to `numpy.sqrt`, for example.
    :param f: function to compute the gradient of
    :param bounds: lower and upper bounds (inclusive) of each coordinate as a list of tuples; to specify exclusive bounds, write `bound - 1e-9` for example
    :param eps: gradient step size
    :return: gradient of `f` as a function of `x`
    """

    def grad_function(x):
        if len(bounds) != len(x):
            raise ValueError('The number of bounds must equal the number of arguments.')

        grad_vector = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] + eps <= bounds[i][1]:
                dx: float = eps
            elif x[i] - eps >= bounds[i][0]:
                dx: float = -eps
            else:
                raise ValueError(f'Gradient cannot be computed because x{i} +- {eps} is out of bounds [{round(bounds[i][0], 3)}, {round(bounds[i][1], 3)}]. '
                                 f'Consider loosening the bounds or decreasing the step size.')
            grad_vector[i] = (f(x + dx) - f(x)) / dx
        return grad_vector

    return grad_function
