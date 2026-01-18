"""
ParticleSystem class for minimd
Holds particle positions, velocities, and masses.
"""
import numpy as np

class ParticleSystem:
    def __init__(self, positions, velocities, masses):
        """
        Initialize the particle system.

        Args:
            positions (np.ndarray): shape (N, 3)
            velocities (np.ndarray): shape (N, 3)
            masses (np.ndarray): shape(N,)
        """
    
        self.positions = np.array(positions, dtype=float)
        self.velocities = np.array(velocities, dtype=float)
        self.masses = np.array(masses, dtype=float)

        # sanity checks
        assert self.positions.shape == self.velocities.shape, "Shape of positions and velocities must match"
        assert self.positions.shape[0] == self.masses.shape[0], "Number of masses instances must match number of particles"

    def num_particles(self):
        """Return the number of particles"""
        return self.positions.shape[0]