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
        input_vec: np.ndarray,
    ) -> np.ndarray:
        """Solve for the state in the next time step.

        Args:
            start_values: The initial values of the variables
            start_time: The time at the start of the step
            time_step: How far ahead to solve for
            input_vec: Input to the derivative function
        """
        raise NotImplementedError
