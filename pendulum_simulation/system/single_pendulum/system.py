import numpy as np

from pendulum_simulation.system.core.abs_system import AbsPhysicsSystem

from .properties import PendulumProperties
from .state import PendulumState


class SimplePendulum(AbsPhysicsSystem[PendulumState, PendulumProperties]):
    """Point mass on a massless rod, theta measured from hanging down.

    θ'' = -(g / L) sin(θ) + τ / (m L^2) - (b / (m L^2)) θ'
    """

    @property
    def state_type(self) -> type[PendulumState]:
        return PendulumState

    def derivatives(
        self,
        time: float,
        state_vec: np.ndarray,
        control_vec: np.ndarray,
    ) -> np.ndarray:
        theta, omega = state_vec
        torque = control_vec[0]

        length = self.properties.length
        mass = self.properties.mass
        damping = self.properties.damping
        g = self.properties.gravity

        alpha = (
            -(g / length) * np.sin(theta)
            + torque / (mass * length**2)
            - damping / (mass * length**2) * omega
        )

        return np.array([omega, alpha])
