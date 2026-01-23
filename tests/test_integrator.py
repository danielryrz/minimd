import numpy as np
from minimd.system import ParticleSystem
from minimd.integrators import EulerIntegrator 

def test_euler_integrator_constant_force():
    """
    A single particle under constant force 
    should accelerate and move in the force directions
    """

    positions = [[0.0, 0.0, 0.0]]
    velocities = [[0.0, 0.0, 0.0]]
    masses = [1.0]

    system = ParticleSystem(positions, velocities, masses)
    integrator = EulerIntegrator()

    force = np.array([[1.0, 0.0, 0.0]]) # constant force in x
    dt = 0.1 

    integrator.step(system, force, dt)

    # After one step:

    # v = a * dt = 1.0 * 0.1 
    assert np.allclose(system.velocities, [[0.1, 0.0, 0.0]]) 

    # x = v * dt = 0.1 * 0.1
    assert np.allclose(system.positions, [[0.01, 0.0, 0.0]] ) 

    # np.allclose() is used to avoid floating point precision issues


