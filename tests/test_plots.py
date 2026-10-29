import pytest

pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")  # no window needed
import matplotlib.pyplot as plt

from pendulum_simulation.core.numerical_solver import RK4Solver  # adjust path
from pendulum_simulation.core.runner import run
from pendulum_simulation.core.simulator import Simulator
from pendulum_simulation.systems.single_pendulum import PendulumState, SimplePendulum
from pendulum_simulation.visualisation.plots import plot_phase, plot_states


@pytest.fixture
def traj():
    sim = Simulator(PendulumState(0.5, 0.0), SimplePendulum(), 0.01, RK4Solver)
    return run(sim, 1.0)


def test_plot_states(traj):
    fig = plot_states(traj)
    assert len(fig.axes) == 2  # theta, omega
    plt.close(fig)


def test_plot_phase_multiple_runs(traj):
    fig = plot_phase([traj, traj], "theta", "omega", labels=["a", "b"])
    assert len(fig.axes[0].lines) == 4  # a line and a start marker per run
    plt.close(fig)


def test_bad_field_raises(traj):
    with pytest.raises(KeyError):
        plot_phase(traj, "theta", "nope")
