from typing import Callable
import numpy as np

from derivatives import jac, hess


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

    # Initialize the search interval and step size:
    a_min: float = 0
    a_max: float = a0
    a: float = (a_max - a_min) / 2

    # Calculate close to optimal step size by binary search until Armijo and curvature conditions are satisfied or the maximum number of iterations has been exceeded:
    n: int = 0
    while n < n_max:
        # Calculate the slope of f at a:
        dfa: float = (f(a + h) - f(a - h)) / (2 * h)
        # If
        # (1) the function value is too high (i.e. above the line f0 + m1 * a * df0) or
        # (2) strong Wolfe line search is enabled and the slope is too positive (i.e. larger than m2 * |df0|),
        # the current step size is used as the right boundary (maximum) for the next iteration:
        if f(a) > f0 + m1 * a * df0 or (strong and dfa > m2 * abs(df0)):
            a_max = a
        # If
        # (3) the slope is too negative (i.e. smaller than m2 * df0),
        # the current step size is used as the left boundary (minimum) for the next iteration:
        elif dfa < m2 * df0:
            a_min = a
        # If
        # (4) the function value is low enough and the slope is neither too high nor too low,
        # exit the loop.
        else:
            break
        # Calculate the next step size by taking the mean of the current search interval:
        a = a_min + (a_max - a_min) / 2
        # Increment the iteration counter:
        n += 1

    # Return step size:
    return a


def convert_to_line(f: Callable[[float], float], position: np.ndarray, direction: np.ndarray) -> Callable[[float], float]:
    """
    Converts a function of many arguments to a function of one argument (i.e. a line) by constraining it to the specified direction.
    """
    return lambda step_size: f(position + step_size * direction)


def l_bfgs_b(f: Callable, g: [Callable], h: [Callable], x0: np.ndarray, max_iterations: float = 1e3, tolerance: float = 1e-3) -> np.ndarray:
    # TODO: Implement side constraints (bounds) via variable projection and subsequent gradient masking.

    # Initialize:
    iteration = 0
    x = x0
    n = len(x)
    m = len(g) + len(h)
    V: np.ndarray = np.eye(n)

    while iteration < max_iterations:
        iteration += 1

        # Calculate A = Jacobian of the constraints:
        A: np.ndarray = jac(f, x)

        # Set up and solve the SQP equation system:
        # noinspection PyTypeChecker
        dim: int = n + m
        LHS = np.zeros(shape=(dim, dim))
        LHS[:n, :n] = V
        LHS[:n, n:] = np.transpose(A)
        LHS[n:, :n] = A
        # noinspection PyTypeChecker
        RHS = np.zeros(dim)
        RHS[:n] = -1 * jac(f, x)
        RHS[n:] = -1 * [gi(x) for gi in g + h]
        result = np.linalg.solve(LHS, RHS)
        p: np.ndarray = result[:n]
        lagrange_multipliers = result[n:]

        # Perform a line search and calculate the updated design variables:
        x_new: np.ndarray = x + line_search(convert_to_line(f, x, p)) * p

        # Calculate the gradient of the Lagrange function at x_new:
        dLdx_new = jac(f, x_new)
        for i, gi in enumerate(g):
            dLdx_new += lagrange_multipliers[i] * jac(gi, x_new)
        for i, hi in enumerate(h):
            dLdx_new += lagrange_multipliers[len(g) + i] * jac(hi, x_new)

        # Check for convergence according to the KKT conditions:
        stationary_point_in_primal_space = (dLdx_new < tolerance * np.ones(n)).all()
        stationary_point_in_dual_space = ([gi(x) for gi in g + h] < tolerance * np.ones(m)).all()
        if stationary_point_in_primal_space or stationary_point_in_dual_space:
            return x_new

        # Calculate the gradient of the Lagrange function at x_new:
        dLdx = jac(f, x_new)
        for i, gi in enumerate(g):
            dLdx += lagrange_multipliers[i] * jac(gi, x_new)
        for i, hi in enumerate(h):
            dLdx += lagrange_multipliers[len(g) + i] * jac(hi, x_new)

        # If not converged, update the Hessian:
        y = dLdx_new - dLdx
        V = (np.eye(n) - np.outer(p, y) / np.inner(y, p)) @ V @ (np.eye(n) - np.outer(y, p) / np.inner(y, p)) + np.outer(p, p) / np.inner(y, p)

        # Update the design variables:
        x = x_new


# For testing:
if __name__ == '__main__':
    pass
