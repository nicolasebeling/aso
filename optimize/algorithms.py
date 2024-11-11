from typing import Callable
import numpy as np

from derivatives import grad, hess


def line_search(f: Callable[[float], float], strong: bool = True, f0: float = None, df0: float = None, h: float = 1e-6, a0: float = 0.9, m1: float = 1e-4, m2: float = 9e-1, n_max: int = 100) -> float:
    """
    Implements a line search algorithm according to the conditions formulated by Larry Armijo and Philip Wolfe.
    :param f: function to be minimized constrained to search direction
    :param strong: whether to use the strong Wolfe line search
    :param f0: initial value of the function
    :param df0: initial value of the function's derivative
    :param h: step size for evaluating the derivative in case it has not been specified
    :param a0: maximum step size guess
    :param m1: constant slope reduction factor used in the Armijo test, must be in (0, 1)
    :param m2: constant slope reduction factor used in the curvature condition, must be in (0, 1)
    :param n_max: maximum number of iterations
    :return: estimated optimal step size `a` in search direction
    """

    # If necessary, calculate initial values of the function and its derivative:
    if f0 is None:
        f0: float = f(0)
    if df0 is None:
        df0: float = (f(h) - f(-h)) / (2 * h)

    # Initialize step_size:
    a_min: float = 0
    a_max: float = a0
    a: float = (a_max - a_min) / 2

    # Calculate close to optimal step size by binary search until Armijo and curvature conditions are satisfied:
    n: int = 0
    while n < n_max:
        dfa: float = (f(a + h) - f(a - h)) / (2 * h)
        if f(a) > f0 + m1 * a * df0 or (strong and dfa > m2 * abs(df0)):
            a_max = a
        elif dfa < m2 * df0:
            a_min = a
        else:
            break
        a = a_min + (a_max - a_min) / 2
        print(a)
        n += 1

    # Return step size:
    return a


def convert_to_line(f: Callable[[float], float], position: np.ndarray, direction: np.ndarray) -> Callable[[float], float]:
    """
    Converts a function of many arguments to a function of one argument (i.e. a line) by constraining it to the specified direction.
    """
    return lambda step_size: f(position + step_size * direction)


def newton(f: Callable[[float], float]):
    pass


def bfgs(f: Callable[[float], float]):
    pass


def minimize_unconstrained(f: Callable[[float], float]):
    pass


def minimize_constrained(f: Callable[[float], float]):
    pass


# For testing:
if __name__ == '__main__':
    def f(x):
        return (x - 5) ** 2


    print(line_search(f))
