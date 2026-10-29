from pendulum_simulation.core.numerical_solver import RK4Solver
from pendulum_simulation.core.runner import run
from pendulum_simulation.core.simulator import Simulator
from pendulum_simulation.systems.single_pendulum import PendulumState, SimplePendulum

system = SimplePendulum(length=1.0)
sim = Simulator(
    initial_state=PendulumState(theta=0.5, omega=0.0),
    system=system,
    control_dt=0.01,
    solver=RK4Solver,
)

trajectory = run(sim, duration=10.0)

theta = trajectory.field("theta")
energies = [
    system.energy(system.state_type.from_array(row)) for row in trajectory.states
]

for i in range(0, len(trajectory.times), 100):
    print(
        f"t={trajectory.times[i]:6.2f} s  "
        f"theta={theta[i]: .4f} rad  "
        f"E={energies[i]:.6f} J"
    )

print(f"energy drift: {max(energies) - min(energies):.2e} J")
