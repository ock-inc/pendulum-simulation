from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np


class AbsSolver(ABC):
    def __init__(
        self,
        derivative: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
        rtol: float,
        atol: float | np.ndarray,
    ):
        """Initialise the solver.

        Args:
            derivative: Function to calculate the current derivative of the state
            rtol: The maximum predicted relative error in `solve`
            atol: The maximum predicted absolute error in `solve`
        """
        self.derivative = derivative
        self.rtol = rtol
        self.atol = atol

    @abstractmethod
    def solve(
        self,
        start_values: np.ndarray,
        start_time: float,
        time_step: float,
        control_vec: np.ndarray,
    ) -> np.ndarray:
        """Solve for the state in the next time step.

        Args:
            start_values: The initial values of the variables
            start_time: The time at the start of the step
            time_step: How far ahead to solve for
        """
        raise NotImplementedError


class EulerSolver(AbsSolver):
    """Basic forward Euler solver"""

    def solve(
        self,
        start_values: np.ndarray,
        start_time: float,
        time_step: float,
        control_vec: np.ndarray,
    ) -> np.ndarray:
        """Estimate the state at the end of one time step using Euler's method."""

        derivatives = self.derivative(
            start_time,
            start_values,
            control_vec,
        )
        return start_values + time_step * derivatives


class RK4Solver(AbsSolver):
    """4th-order Runge-Kutta solver"""

    def solve(
        self,
        start_values: np.ndarray,
        start_time: float,
        time_step: float,
        control_vec: np.ndarray,
    ) -> np.ndarray:
        """Advance the state by `time_step`, holding `control_vec` constant."""
        f = self.derivative
        h = time_step
        t = start_time
        y = np.asarray(start_values, dtype=float)

        k1 = np.asarray(f(t, y, control_vec), dtype=float)
        k2 = np.asarray(f(t + h / 2, y + h / 2 * k1, control_vec), dtype=float)
        k3 = np.asarray(f(t + h / 2, y + h / 2 * k2, control_vec), dtype=float)
        k4 = np.asarray(f(t + h, y + h * k3, control_vec), dtype=float)

        return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
