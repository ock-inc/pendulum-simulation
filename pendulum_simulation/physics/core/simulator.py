from __future__ import annotations

import itertools
from collections.abc import Callable, Generator

import numpy as np

from pendulum_simulation.system.core.abs_state import AbsState
from pendulum_simulation.system.core.abs_system import AbsPhysicsSystem

from .numerical_solver import AbsSolver


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
        self.control = np.zeros(system.control_dim)
        self.solver = solver(system.derivatives, rtol, atol)

        self._constraints = list(system.constraints())

    def step(self, control_vec: np.ndarray | float | None = None) -> S:
        """Take a single step in the simulation.

        Args:
            control_vec: The value of the inputs to control the system
                defaults to zeros for a system with no inputs
        """
        if control_vec is None:
            control_vec = np.zeros(self.system.control_dim, dtype=float)
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
        self.control = control_vec

        return self.state

    def run(
        self,
        duration: float | None = None,
        controller: Callable[[S], np.ndarray] | None = None,
    ) -> Generator[S]:
        """Step repeatedly, yielding the result of each step as it happens.

        Args:
            duration: How long to run for, in whole control steps. None runs forever.
            controller: Chooses the control from the current state each step.
                If omitted, zero control is applied.
        """

        if duration is None:
            duration = np.inf

        for step in itertools.count():
            if step * self.control_dt > duration:
                break
            control = controller(self.state) if controller else None
            yield self.step(control)
