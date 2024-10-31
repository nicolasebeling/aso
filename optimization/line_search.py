from typing import Callable


def line_search(f: Callable[[float], float]) -> float:
    """
    :param f: function to be minimized constrained to search direction
    :return: step size in search direction
    """
    pass
