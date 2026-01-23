"""
Integrator implementation for minimd. 
"""
import numpy as np

class EulerIntegrator:
    """
    Explicit Euler time integration - used as a starting intergrator for simplicity and testing.
    Adavntages: simple to implement, computationally inexpensive (first order accurate).
    Disadvantages: poor energy conservation, stable only for MD with small dt.
    How it works: 
    1) based on acting Force, finds current accelerations 
    2) updates next step velocities using current accelerations.
    3) updates next step positions using next step velocities. (semi-implicit as using next step velocities)
    """

    def step(self, system, forces, dt):
        """
        Advance the system by one time step using Euler integration 

        Args:
            system (ParticleSystem): system to update
            forces (np.ndarray): shape (N,3)
            dt (float): time step 
        """
        
        accelerations = forces / system.masses[:, np.newaxis]

        system.velocities += accelerations * dt 
        system.positions += system.velocities * dt


