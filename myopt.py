from typing import Callable


def gradient_descent(f: Callable[[float, ...], float], x0: tuple[float]):
    pass


def line_search(f: Callable[[float], float]) -> float:
    """
    :param f: function to be minimized constrained to search direction
    :return: step size in search direction
    """
    pass
