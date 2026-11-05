from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .abs_state import AbsState
from .abs_system_properties import AbsSystemProperties


class AbsPhysicsSystem[S: AbsState, P: AbsSystemProperties](ABC):
    """Abstract physics: equations of motion, state type, actuator limits, constraints."""

    def __init__(self, properties: P):
        self.properties = properties

    @property
    @abstractmethod
    def state_type(self) -> type[S]:
        """Return the state type used to represent the system's state."""
        raise NotImplementedError

    @abstractmethod
    def derivatives(
        self,
        time: float,
        state_vec: np.ndarray,
        control_vec: np.ndarray,
    ) -> np.ndarray:
        """dy/dt for the state vector y under a control input.

        Args:
            time: Current time.
            state_vec: State as an array, in the field order of `state_type`.
            control_vec: Control inputs of shape `(control_dim,)`, held constant over the step.
        """
        raise NotImplementedError

    @property
    def control_dim(self) -> int:
        """The number of control variables."""
        return self.properties.control_bounds[0].size
