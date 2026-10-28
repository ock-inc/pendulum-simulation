import numpy as np

from pendulum_simulation.core.numerical_solver import AbsSolver, EulerSolver, RK4Solver
from pendulum_simulation.core.simulator import Simulator
from pendulum_simulation.systems.single_pendulum import PendulumState, SimplePendulum


def energy_drift(solver: type[AbsSolver], steps: int = 1000, dt: float = 0.01) -> float:
    """Max relative deviation of total energy from its initial value."""
    system = SimplePendulum(length=1.0)
    sim = Simulator(
        initial_state=PendulumState(theta=0.5, omega=0.0),
        system=system,
        control_dt=dt,
        solver=solver,
    )

    initial = system.energy(sim.state)
    energies = [initial]
    for _ in range(steps):
        sim.step()
        energies.append(system.energy(sim.state))

    return float(np.max(np.abs(np.array(energies) - initial)) / initial)


if __name__ == "__main__":
    for solver in (EulerSolver, RK4Solver):
        print(f"{solver.__name__:>12}: relative energy drift = {energy_drift(solver):.2e}")