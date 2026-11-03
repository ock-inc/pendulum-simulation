import numpy as np

from pendulum_simulation.core.numerical_solver import AbsSolver


class RK4Solver(AbsSolver):
    """4th-order Runge-Kutta solver"""

    def solve(
        self,
        start_values: np.ndarray,
        start_time: float,
        time_step: float,
        input_vec: np.ndarray,
    ) -> np.ndarray:
        """Advance the state by `time_step`, holding `input_vec` constant."""
        f = self.derivative
        h = time_step
        t = start_time
        y = np.asarray(start_values, dtype=float)

        k1 = np.asarray(f(t, y, input_vec), dtype=float)
        k2 = np.asarray(f(t + h / 2, y + h / 2 * k1, input_vec), dtype=float)
        k3 = np.asarray(f(t + h / 2, y + h / 2 * k2, input_vec), dtype=float)
        k4 = np.asarray(f(t + h, y + h * k3, input_vec), dtype=float)

        return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
