from minimd.system import ParticleSystem

def test_kinetic_energy_zero_velocity():
    """
    Test that kinetic energy is zero when all velocities are zero.
    """

    
    positions = [[0.0, 0.0, 0.0]]
    velocities = [[0.0, 0.0, 0.0]]
    masses = [2.0]
 
    system = ParticleSystem(positions, velocities, masses)

    assert system.kinetic_energy() == 0.0, "Kinetic energy should be zero for zero velocities"

def test_lj_potential_minimum():
    """
    Test that Lennard-Jones potential is minimum at r = 2^(1/6) * sigma, potential is -epsilon.
    """
    import numpy as np
    from minimd.potential_energy import lennard_jones_potential

    sigma = 1.0
    r_min = 2**(1/6) * sigma

    positions = np.array([[0.0, 0.0, 0.0],
                          [r_min, 0.0, 0.0]])
    
    potential = lennard_jones_potential(positions, epsilon=1.0, sigma=sigma)

    expected_min_potential = -1.0  # Minimum potential value for LJ potential with epsilon=1.0

    assert np.isclose(potential, expected_min_potential, atol=1e-5), f"LJ potential should be minimum at r={r_min}"