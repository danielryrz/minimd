import numpy as np
from minimd.forces import lennard_jones_forces

def test_lj_force_symmetry():
    """
    Test that Lennard-Jones forces are equal and opposite for a pair of particles.
    """
    positions = np.array([[0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0]]) # Two particles 1 unit apart


    forces = lennard_jones_forces(positions)

    # Forces on particle 0 and 1 should be equal and opposite: Newton's 3rd law
    assert np.allclose(forces[0], -forces[1]), "Forces are not equal and opposite"

def test_lj_force_zero_far_away():
    """
    Test that Lennard-Jones forces approach zero as particles are far apart.
    """
    positions = np.array([[0.0, 0.0, 0.0],
                          [100.0, 0.0, 0.0]]) # Two particles far apart

    forces = lennard_jones_forces(positions)

    # Forces should be very small when particles are far apart
    assert np.linalg.norm(forces) < 1e-6, "Forces are not negligible for distant particles"