from __future__ import annotations

from abc import ABC
from dataclasses import astuple, dataclass, fields
from typing import ClassVar, Self

import numpy as np


@dataclass(frozen=True)
class AbsState(ABC):
    """Base for all states. Subclasses only declare float fields."""

    ANGLE_FIELDS: ClassVar[tuple[str, ...]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if "ANGLE_FIELDS" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define ANGLE_FIELDS")

    def as_array(self) -> np.ndarray:
        """Return the state as a numpy array."""
        return np.array(astuple(self), dtype=float)

    @classmethod
    def from_array(cls, array: np.ndarray) -> Self:
        """Create a `State` from a numpy array"""
        values = np.asarray(array, dtype=float).ravel()

        state_fields = fields(cls)
        if len(values) != len(state_fields):
            raise ValueError(
                f"Expected {len(state_fields)} values for {cls.__name__}, "
                f"but got {len(values)}"
            )

        return cls(*values)

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """The names of fields defining the state."""
        return tuple(f.name for f in fields(cls))

    @property
    def energy(self) -> float:
        """Total energy in the given state."""
        raise NotImplementedError
