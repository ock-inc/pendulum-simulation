from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .controller import AbsController
    from .simulator import Simulator


@dataclass(frozen=True)
class Trajectory:
    """The recorded result of a simulation run.

    Row `i` of `states` is the state at `times[i]`. Row `i` of `controls` is
    the control applied between `times[i]` and `times[i + 1]`, so there is one
    fewer control than there are states.
    """

    times: np.ndarray  # shape (N + 1,)
    states: np.ndarray  # shape (N + 1, state_dim)
    controls: np.ndarray  # shape (N, control_dim)
    field_names: tuple[str, ...]

    def __post_init__(self) -> None:
        n = len(self.times)
        if self.states.shape != (n, len(self.field_names)):
            raise ValueError(
                f"states has shape {self.states.shape}, expected "
                f"{(n, len(self.field_names))}"
            )
        if len(self.controls) != n - 1:
            raise ValueError(
                f"controls has {len(self.controls)} rows, expected {n - 1}"
            )

    def field(self, name: str) -> np.ndarray:
        """The values of one state field over time, shape (N + 1,)."""
        try:
            index = self.field_names.index(name)
        except ValueError:
            raise KeyError(
                f"No state field '{name}'; available: {self.field_names}"
            ) from None
        return self.states[:, index]


def run(
    simulator: Simulator,
    duration: float,
    controller: AbsController | None = None,
) -> Trajectory:
    """Step the simulator for `duration` seconds and record every step.

    Args:
        simulator: The simulator to advance. Its current state is the start.
        duration: How long to run for. Must be a whole number of control steps.
        controller: Chooses the control at each step from the current state.
            If omitted, zero control is applied.
    """
    dt = simulator.control_dt
    n_steps = round(duration / dt)

    if n_steps < 0 or not math.isclose(
        n_steps * dt, duration, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(
            f"duration {duration} is not a non-negative multiple of control_dt {dt}"
        )

    control_dim = simulator.system.control_dim
    field_names = simulator.system.state_type.field_names()

    times = np.empty(n_steps + 1)
    states = np.empty((n_steps + 1, len(field_names)))
    controls = np.empty((n_steps, control_dim))

    times[0] = simulator.time
    states[0] = simulator.state.as_array()

    for i in range(n_steps):
        if controller is None:
            control = np.zeros(control_dim)
        else:
            control = controller(simulator.state)

        state = simulator.step(control)

        controls[i] = control
        times[i + 1] = simulator.time
        states[i + 1] = state.as_array()

    return Trajectory(times, states, controls, field_names)
