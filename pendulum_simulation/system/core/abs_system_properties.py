from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .abs_state import AbsState
from .constraint import Constraint
from .exceptions import InvalidStateError


@dataclass(frozen=True, kw_only=True)
class AbsSystemProperties[S: AbsState](ABC):
    """Base for all system parameters."""

    control_bounds: tuple[np.ndarray, np.ndarray]
    constraints: Sequence[Constraint] = ()

    def __post_init__(self):
        if self.control_bounds[0].shape != self.control_bounds[1].shape:
            raise ValueError(
                "control bounds has inconsistent length for max and min values"
            )

    def validate(self, state: S) -> None:
        """Validate the given state doesn't violate any constraints."""
        for c in self.constraints:
            if not c.fn(state):
                raise InvalidStateError(
                    f"{c.field} violates constraint '{c.name}'", state, c.name
                )

    @abstractmethod
    def energy(self, state: S) -> float:
        raise NotImplementedError
