
from dataclasses import dataclass

from pendulum_simulation.core.state import AbsState


@dataclass(frozen=True)
class PendulumState(AbsState):
    """Angle from the downward vertical (rad) and angular velocity (rad/s)."""

    ANGLE_FIELDS = ("theta",)

    theta: float
    omega: float