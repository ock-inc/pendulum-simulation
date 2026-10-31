from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pendulum_simulation.core.state import AbsState
from pendulum_simulation.core.system import AbsPhysicsSystem


class AbsAutomatedController[S: AbsState](ABC):
    """Maps a state to a control vector, clipped to the system's actuator bounds."""

    def __init__(self, system: AbsPhysicsSystem[S]):
        self.low, self.high = system.control_bounds

    def __call__(self, state: S) -> np.ndarray:
        u = np.atleast_1d(np.asarray(self.compute_control(state), dtype=float))

        return np.clip(u, self.low, self.high)

    @abstractmethod
    def compute_control(self, state: S) -> np.ndarray:
        """Calculate the control input for the system in the given state."""
        raise NotImplementedError
