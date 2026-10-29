from __future__ import annotations

import numpy as np

from .numerical_solver import AbsSolver
from .state import AbsState
from .system import AbsPhysicsSystem


class Simulator[S: AbsState]:
    """Integrates any PhysicsSystem over one control interval with a specified solver."""

    def __init__(
        self,
        initial_state: S,
        system: AbsPhysicsSystem[S],
        control_dt: float,
        solver: type[AbsSolver],
        rtol: float = 1e-6,
        atol: float | np.ndarray = 1e-8,
    ):
        """Initialise the simulator.

        Args:
            initial_state: The starting state for the simulation
            system: The definition of how the system behaves
            control_dt: The timestep to take between simulation steps
            solver: The numerical solver to calculate the next state
            rtol: The maximum predicted relative error of one simulation step
            atol: The maximum predicted absolute error of one simulation step
        """
        self.state = initial_state
        self.system = system
        self.control_dt = control_dt
        self.rtol = rtol
        self.atol = atol

        self.time = 0.0
        self.solver = solver(system.derivatives, rtol, atol)

        self._constraints = list(system.constraints())

    def step(self, control_vec: np.ndarray | float | None = None) -> S:
        """Take a single step in the simulation.

        Args:
            control_vec: The value of the inputs to control the system
                defaults to zeros for a system with no inputs
        """
        if control_vec is None:
            control_vec = np.zeros(self.system.control_dim)
        else:
            control_vec = np.atleast_1d(np.asarray(control_vec, dtype=float))

        if control_vec.shape != (self.system.control_dim,):
            raise ValueError(
                f"Expected control of shape ({self.system.control_dim},), got {control_vec.shape}"
            )

        new_state_vec = self.solver.solve(
            self.state.as_array(),
            self.time,
            self.control_dt,
            control_vec,
        )
        new_state = self.system.state_type.from_array(new_state_vec)

        self.system.validate(new_state)

        self.state = new_state
        self.time += self.control_dt

        return self.state
