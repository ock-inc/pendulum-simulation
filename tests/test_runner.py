import numpy as np
import pytest

from pendulum_simulation.core.numerical_solver import RK4Solver
from pendulum_simulation.core.runner import run
from pendulum_simulation.core.simulator import Simulator
from pendulum_simulation.systems.single_pendulum import PendulumState, SimplePendulum


@pytest.fixture
def sim():
    return Simulator(PendulumState(0.5, 0.0), SimplePendulum(), 0.01, RK4Solver)


def test_shapes_and_endpoints(sim):
    traj = run(sim, 1.0)
    assert traj.times.shape == (101,)
    assert traj.states.shape == (101, 2)
    assert traj.controls.shape == (100, sim.system.control_dim)
    assert traj.times[0] == 0.0
    assert traj.times[-1] == pytest.approx(1.0)
    np.testing.assert_array_equal(traj.states[0], [0.5, 0.0])


def test_field_lookup(sim):
    traj = run(sim, 0.1)
    np.testing.assert_array_equal(traj.field("theta"), traj.states[:, 0])
    with pytest.raises(KeyError):
        traj.field("nope")


def test_rejects_non_multiple_duration(sim):
    with pytest.raises(ValueError):
        run(sim, 1.005)
