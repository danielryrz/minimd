"""
Placeholder for Simulation loop.
"""
class Simulation:
    def __init__(self, system, integrator):
        self.system = system
        self.integrator = integrator

    def run(self, steps, dt):
        for _ in range(steps):
            # TODO: compute forces and step system
            pass
