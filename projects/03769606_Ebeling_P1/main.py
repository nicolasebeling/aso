from optimizer import Optimizer
import numpy as np

# Define the start point here
start_point = np.array([2, 2])


def main(start_point: np.ndarray):
    start_point = start_point
    optimisation_problem = Optimizer(start_point=start_point,
                                     objective_function="ROSENBROCK",
                                     search_direction_algorithm="STEEPEST_DESCENT",
                                     max_iterations=1e4)
    solution = optimisation_problem.optimize()
    print(solution)


if __name__ == "__main__":
    main(start_point)
