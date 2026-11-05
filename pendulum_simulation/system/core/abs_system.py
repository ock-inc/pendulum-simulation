from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from .abs_state import AbsState
from .constraint import Constraint
from .exceptions import InvalidStateError


class AbsPhysicsSystem[S: AbsState](ABC):
    """Abstract physics: equations of motion, state type, actuator limits, constraints."""

    @property
    @abstractmethod
    def state_type(self) -> type[S]:
        """Return the state type used to represent the system's state."""
        raise NotImplementedError

    @property
    @abstractmethod
    def control_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(low, high) arrays, each of shape (control_dim,).

        Limits of the inputs used to control the system.
        """
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

    def constraints(self) -> Sequence[Constraint]:
        """The constraints for the system."""
        return ()

    @property
    def control_dim(self) -> int:
        """The number of control variables."""
        return self.control_bounds[0].size

    def validate(self, state: S) -> None:
        """Validate the given state doesn't violate any constraints."""
        for c in self.constraints():
            if not c.fn(state):
                raise InvalidStateError(
                    f"{c.field} violates constraint '{c.name}'", state, c.name
                )

    @abstractmethod
    def energy(self, state: S) -> float:
        """Total energy in the given state."""
        raise NotImplementedError
