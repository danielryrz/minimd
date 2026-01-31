import numpy as np
from minimd.system import ParticleSystem
from minimd.integrators import EulerIntegrator, VelocityVerletIntegrator

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

def test_velocity_verlet_zero_force_constant_velocity():
    """
    A single particle with zero forces should maintain constant velocity
    using Velocity Verlet integrator.
    """

    positions = [[0.0, 0.0, 0.0]]
    velocities = [[1.0, 0.0, 0.0]]  # initial velocity in x
    masses = [1.0]

    system = ParticleSystem(positions, velocities, masses)
    integrator = VelocityVerletIntegrator()

    force = np.array([[0.0, 0.0, 0.0]]) # zero force

    def zero_force_fn(pos):
        return np.zeros_like(pos)
    
    forces = zero_force_fn(system.positions)
    dt = 0.1 

    integrator.step(system, forces, dt, zero_force_fn)

    # After one step:

    # velocity should remain unchanged
    assert np.allclose(system.velocities, [[1.0, 0.0, 0.0]]) # allclose due to floating rounding error

    # position should update based on constant velocity (i.e. should advance linearly)
    # As the x direction: x(0 + 0.1) = x(0) + v(0) * 0.1 + 1/2 * a(0) * (0.1**2)
    # Gives: x(0.1) = 0 + 1.0 * 0.1 + * 0
    # Gives: x(0.1) = 0.1
    # y(0.1) = 0 and z(0.1) = 0, as v_y(0) = 0 and v_z(0) = 0 
    assert np.allclose(system.positions, [[0.1, 0.0, 0.0]] )
