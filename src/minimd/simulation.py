"""
Placeholder for Simulation loop.
"""
from minimd.forces import lennard_jones_forces

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
