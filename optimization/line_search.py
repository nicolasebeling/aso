from typing import Callable
import numpy as np


def line_search(f: Callable[[float], float], f0: float = None, df0: float = None, h: float = 1e-3, algorithm: str = 'armijo', a0: float = 0.1) -> float:
    """
    Implements the line search algorithms according to Larry Armijo, Jerome Goldstein and Philip Wolfe.
    :param f: function to be minimized constrained to search direction
    :param f0: initial value of the function
    :param df0: initial value of the derivative
    :param h: step size for evaluating the derivative in case it has not been specified
    :param algorithm: line search algorithm
    :param a0: initial step size
    :return: estimated optimal step size `a` in search direction
    """
    if f0 is None:
        f0 = f(0)
    if df0 is None:
        df0 = (f(h) - f(-h)) / (2 * h)
    a: float = a0
    m = 1e-4  # constant slope reduction factor in (0, 1))
    b: float = 0.5  # constant backtracking factor in (0, 1)
    while f(a) > f0 + m * a * df0:
        a = b * a
    return a


def convert_to_line(f: Callable[[float], float], position: np.ndarray, direction: np.ndarray) -> Callable[[float], float]:
    """
    Converts a function of many arguments to a function of one argument (a line) by constraining it to the specified direction.
    :param f:
    :param position:
    :param direction:
    :return:
    """
    return lambda step_size: f(position + step_size * direction)


# For testing:
if __name__ == '__main__':
    pass
