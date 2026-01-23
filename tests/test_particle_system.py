import numpy as np
from minimd.system import ParticleSystem

def test_particle_system_shapes():
    positions = [[0,0,0], [1,1,1]]
    velocities = [[0,0,0], [0,0,0]]
    masses = [1.0, 2.0]

    system = ParticleSystem(positions, velocities, masses)
    assert system.num_particles() == 2
    assert system.positions.shape == (2,3)
    assert system.velocities.shape == (2,3)
    assert system.masses.shape == (2,)
