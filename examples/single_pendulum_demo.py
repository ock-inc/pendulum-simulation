from pendulum_simulation.core.numerical_solver import RK4Solver
from pendulum_simulation.core.simulator import Simulator
from pendulum_simulation.systems.single_pendulum import PendulumState, SimplePendulum

system = SimplePendulum(length=1.0)
sim = Simulator(
    initial_state=PendulumState(theta=0.5, omega=0.0),
    system=system,
    control_dt=0.01,
    solver=RK4Solver,
)

energies = [system.energy(sim.state)]
for _ in range(1000):
    sim.step()
    energies.append(system.energy(sim.state))

print(f"energy drift: {max(energies) - min(energies):.2e}")