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

times, thetas, energies = [sim.time], [sim.state.theta], [system.energy(sim.state)]

for _ in range(1000):
    sim.step()
    times.append(sim.time)
    thetas.append(sim.state.theta)
    energies.append(system.energy(sim.state))

for i in range(0, len(times), 100):
    print(f"t={times[i]:6.2f}  theta={thetas[i]: .4f}  E={energies[i]:.6f}")

print(f"energy drift: {max(energies) - min(energies):.2e}")