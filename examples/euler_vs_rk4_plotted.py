import matplotlib.pyplot as plt

from pendulum_simulation.core.numerical_solver import EulerSolver, RK4Solver
from pendulum_simulation.core.runner import run
from pendulum_simulation.core.simulator import Simulator
from pendulum_simulation.systems.single_pendulum import PendulumState, SimplePendulum
from pendulum_simulation.visualisation.plots import plot_phase, plot_states

trajectories = []
for solver in (EulerSolver, RK4Solver):
    sim = Simulator(PendulumState(0.5, 0.0), SimplePendulum(), 0.01, solver)
    trajectories.append(run(sim, duration=10.0))

labels = ["Euler", "RK4"]
plot_states(trajectories, labels)
plot_phase(trajectories, "theta", "omega", labels)
plt.show()