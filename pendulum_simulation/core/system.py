from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np

from .state import AbsState


@dataclass(frozen=True)
class Constraint:
    """Validity condition on a state: `fn(state) == True` means valid."""

    name: str
    field: str
    fn: Callable[[AbsState], bool]

    @classmethod
    def get_constraint(
        cls,
        name: str,
        field: str,
        minimum: float = -float("inf"),
        maximum: float = float("inf"),
    ) -> Self:
        """Create a constraint for a value on a state.

        Args:
            name: An identifier for the constraint
            field: The name of the field on the state that the constraint is for
            minimum: The minimum allowable value of the field
            maximum: The maximum allowable value of the field
        """

        def _check(state: AbsState) -> bool:
            if field not in state.field_names():
                raise ValueError(
                    f"Field {field} not present on {type(state).__name__} state"
                )

            value = getattr(state, field)

            return minimum <= value <= maximum

        return cls(name, field, _check)


class InvalidStateError(Exception):
    def __init__(self, message: str, state: AbsState, constraint: str):
        super().__init__(message)
        self.state = state
        self.constraint = constraint


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
            time: Current time in seconds.
            state_vec: State as an array, in the field order of `state_type`.
            control_vec: Control inputs of shape `(control_dim,)`, held constant over the step.
        """
        raise NotImplementedError

    @abstractmethod
    def energy(self, state: S) -> float:
        """Total mechanical energy (kinetic + potential) in the given state."""
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
