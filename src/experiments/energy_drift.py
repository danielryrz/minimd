import numpy as np
from minimd.system import ParticleSystem
from minimd.forces import lennard_jones_forces
from minimd.integrators import EulerIntegrator, VelocityVerletIntegrator
from minimd.simulation import Simulation
from minimd.energy import total_energy 

print("Running energy drift experiment...")

positions = [[0,0,0], [1.1, 0, 0]]
velocities = [[0,0,0], [0,0.1,0]]
masses = [1.0, 1.0]

dt = 0.005
steps = 2000

def run(integrator):
    system = ParticleSystem(positions, velocities, masses)
    sim = Simulation(system, integrator)
    energies = []

    for _ in range(steps):
        energies.append(total_energy(system))
        sim.run(1, dt)

    return np.array(energies)

euler_E = run(EulerIntegrator())
verlet_E = run(VelocityVerletIntegrator())

print("Euler energy drift:", euler_E[-1] - euler_E[0])
print("Verlet energy drift:", verlet_E[-1] - verlet_E[0])
