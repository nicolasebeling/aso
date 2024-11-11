from objective_functions.objective_functions import *
from search_direction.search_direction_algorithms import *
import numpy as np


class Optimizer():
    """This class initializes an Optimizer object (i.e. creates an optimize problem). The optimisation problem is solved with the optimize method.
    Args:
        start_point (np.ndarray): Input vector for the chosen objective function
        objective_function (str): Choose an objective function to test your optimize algorithm ('ROSENBROCK')
        search_direction_algorithm (str): Choose an algorithm to determine the search direction ('STEEPEST DESCENT')
        line_search_algorithm (str): Choose an algorithm to determine alpha (not implemented yet, alpha defaults to 0.3e-4)
        tolerance (float): Define the tolerance for your result. This will be compared to ||∇f(x)||_{inf}, which is the maximum absolute value of the gradient at the current point
        max_iterations (int): Set a value for the maximum amount of iterations. When this is reached by the counter, the optimize will stop
    """

    def __init__(self,
                 start_point: np.ndarray,
                 objective_function: str = "ROSENBROCK",
                 search_direction_algorithm: str = "STEEPEST_DESCENT",
                 line_search_algorithm: str = ' ',
                 tolerance: float = 1e-3,
                 max_iterations: int = 2000
                 ):

        self.current_point = start_point
        self.tolerance = tolerance
        self.iteration = 0
        self.max_iterations = max_iterations
        self.converged = False

        if line_search_algorithm == ' ':
            self.alpha = 3e-4

        if objective_function == "ROSENBROCK":
            print("Initialize rosenbrock objective function...")
            self.objective_function = Rosenbrock(self.current_point)
            self.gradient = self.objective_function.compute_gradient_at_x()
            self.max_gradient_value = np.max(abs(self.gradient))

        if search_direction_algorithm == "STEEPEST_DESCENT":
            print("Search direction algorithm: Steepest Descent...")
            self.search_direction_normalized = SteepestDescent.compute_search_direction(self.gradient)

    # Implement the optimize algorithm here. Do not change anything above this line.
    def optimize(self):
        # While the optimality condition is not fulfilled and the number of iterations is lower than the specified maximum,
        while np.max(np.abs(self.gradient)) > self.tolerance and self.iteration < self.max_iterations:
            # ...take a step of length alpha in the opposite direction of the gradient,
            self.current_point = self.current_point + self.alpha * self.search_direction_normalized
            # ...update the objective function,
            self.objective_function.x = self.current_point
            # ...update the gradient,
            self.gradient = self.objective_function.compute_gradient_at_x()
            # ...compute the search direction for the next step,
            self.search_direction_normalized = SteepestDescent.compute_search_direction(self.gradient)
            # ...and increment the iteration counter.
            self.iteration += 1

        # Return the final point (ideally the optimum), the final value of the objective function and the final number of iterations as a string.
        return (f'Optimization finished at the following point: \n'
                f'x*        = {self.current_point} \n'
                f'f(x*)     = {self.objective_function.compute_value_at_x()} \n'
                f'k         = {self.iteration}')
