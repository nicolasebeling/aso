from typing import Callable

import numpy as np


def create_grad_within_bounds(f: Callable[[np.ndarray], float], bounds: list[tuple], h: float = 1e-8) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a function that numerically computes the gradient of the function `f` as a function of `x`. If bounds are specified (obligatory in the current implementation), they are not exceeded.
    This is useful if `f` is cannot be computed outside these bounds, due to negative arguments being passed to `numpy.sqrt`, for example.
    :param f: function to compute the gradient of
    :param bounds: lower and upper bounds (inclusive) of each coordinate as a list of tuples; to specify exclusive bounds, write `bound - 1e-9` for example
    :param h: step size
    :return: gradient of `f` as a function of `x`
    """

    def calculate_grad_within_bounds(x):
        if len(bounds) != len(x):
            raise ValueError('The number of bounds must equal the number of arguments.')

        grad = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] + h <= bounds[i][1]:
                dx: float = h
            elif x[i] - h >= bounds[i][0]:
                dx: float = -h
            else:
                raise ValueError(f'Gradient cannot be computed because x{i} +- {h} is out of bounds [{round(bounds[i][0], 3)}, {round(bounds[i][1], 3)}]. '
                                 f'Consider loosening the bounds or decreasing the step size.')
            grad[i] = (f(x + dx) - f(x)) / dx
        return grad

    return calculate_grad_within_bounds


def calculate_grad(f: Callable[[float], float], x: [float], h: float = 1e-3) -> np.ndarray:
    """
    Evaluates the gradient of the function 'f' numerically using the central difference method.
    :param f: function to compute the gradient of
    :param x: input vector
    :param h: step size
    :return: gradient of `f` at `x`
    """
    grad = np.zeros(len(x))
    for i in range(len(x)):
        h_vector = np.zeros_like(x, dtype=float)
        h_vector[i] = h
        grad[i] = (f(x + h_vector) - f(x - h_vector)) / (2 * h)
    return grad


def calculate_hess(f: Callable[[float], float], x: [float], h: float = 1e-3) -> np.ndarray:
    """
    Evaluates the hessian of the function 'f' numerically using the central difference method.
    :param f: function to compute the gradient of
    :param x: input vector
    :param h: step size
    :return: hessian of `f` at `x`
    """
    hess = np.zeros(shape=(len(x), len(x)))
    for i in range(len(x)):
        def calculate_grad_i(y: [float]) -> float:
            return float(calculate_grad(f, y, h)[i])

        hess[i] = calculate_grad(calculate_grad_i, x, h)
    return hess


# For testing:
if __name__ == '__main__':
    pass
