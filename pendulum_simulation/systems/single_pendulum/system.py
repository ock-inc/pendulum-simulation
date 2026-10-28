import numpy as np

from pendulum_simulation.core.system import AbsPhysicsSystem

from .state import PendulumState


class SimplePendulum(AbsPhysicsSystem[PendulumState]):
    """Point mass on a massless rod, theta measured from hanging down.

    theta'' = -(g / L) sin(theta) - (b / (m L^2)) theta'
    """

    def __init__(
        self,
        length: float = 1.0,
        mass: float = 1.0,
        damping: float = 0.0,
        gravity: float = 9.81,
    ):
        if length <= 0 or mass <= 0:
            raise ValueError("length and mass must be positive")
        if damping < 0:
            raise ValueError("damping must be non-negative")

        self.length = length
        self.mass = mass
        self.damping = damping
        self.gravity = gravity

    @property
    def state_type(self) -> type[PendulumState]:
        return PendulumState

    @property
    def control_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        # Minimum and maximum applied torque, in N m.
        return (
            np.array([-10.0]),
            np.array([10.0]),
        )

    def derivatives(
        self,
        time: float,
        state_vec: np.ndarray,
        control_vec: np.ndarray,
    ) -> np.ndarray:
        theta, omega = state_vec
        torque = control_vec[0]

        alpha = (
            -(self.gravity / self.length) * np.sin(theta)
            + torque / (self.mass * self.length**2)
            - self.damping / (self.mass * self.length**2) * omega
        )

        return np.array([omega, alpha])

    def energy(self, state: PendulumState) -> float:
        """Total mechanical energy, with zero potential at the lowest point."""
        kinetic = 0.5 * self.mass * (self.length * state.omega) ** 2
        potential = self.mass * self.gravity * self.length * (1 - np.cos(state.theta))
        return kinetic + potential