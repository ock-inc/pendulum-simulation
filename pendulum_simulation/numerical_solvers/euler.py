import numpy as np

from pendulum_simulation.core.numerical_solver import AbsSolver


class EulerSolver(AbsSolver):
    """Basic forward Euler solver"""

    def solve(
        self,
        start_values: np.ndarray,
        start_time: float,
        time_step: float,
        input_vec: np.ndarray,
    ) -> np.ndarray:
        """Estimate the state at the end of one time step using Euler's method."""

        derivatives = self.derivative(
            start_time,
            start_values,
            input_vec,
        )
        return start_values + time_step * derivatives
