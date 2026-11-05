from dataclasses import dataclass

import numpy as np

from ..core.abs_system_properties import AbsSystemProperties
from .state import PendulumState


@dataclass(frozen=True, kw_only=True)
class PendulumProperties(AbsSystemProperties[PendulumState]):
    """Base for all system parameters."""

    length: float
    mass: float
    damping: float
    gravity: float

    def __post_init__(self):
        super().__post_init__()
        if self.length <= 0:
            raise ValueError(f"Pendulum length must be positive, not {self.length!r}")
        if self.mass <= 0:
            raise ValueError(f"Pendulum mass must be positive, not {self.mass!r}")
        if self.damping < 0:
            raise ValueError(f"Pendulum damping must be positive, not {self.damping!r}")

    def energy(self, state: PendulumState) -> float:
        kinetic = 0.5 * self.mass * (self.length * state.omega) ** 2
        potential = self.mass * self.gravity * self.length * (1 - np.cos(state.theta))
        return kinetic + potential
