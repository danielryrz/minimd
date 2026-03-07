"""
Placeholder for Simulation loop.
"""
from minimd.forces import lennard_jones_forces
from minimd.potential_energy import lennard_jones_potential

class Simulation:
    def __init__(self, system, integrator, force_fn=lennard_jones_forces):
        self.system = system
        self.integrator = integrator
        self.force_fn = force_fn

    def run(self, steps, dt):
        # initialize forces
        forces = self.force_fn(self.system.positions)

        for _ in range(steps):
            forces = self.integrator.step(self.system, forces, dt, self.force_fn)

    # moved to energy.py
    # def total_energy(system):
    #     kinetic = system.kinetic_energy()
    #     potential = lennard_jones_potential(system.positions)
    #     return kinetic + potential
    
    
