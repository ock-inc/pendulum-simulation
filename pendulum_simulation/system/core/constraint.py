from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

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
