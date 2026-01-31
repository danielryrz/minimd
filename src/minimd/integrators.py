"""
Integrator implementation for minimd. 
"""
import numpy as np

from minimd.forces import lennard_jones_forces

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


class VelocityVerletIntegrator:
    """
    Velocity Verlet time integration - commonly used in MD simulations. 
    Advantages: good energy conservation, time-reversible, symplectic (preserves phase space volume).
    Disadvantages: slightly more computationally expensive than Euler (second order accurate).
    Overall balances efficiency with energy conservation well for MD.
    How it works:
    1) based on acting Force, finds current accelerations 
    2) updates next step positions using current velocities and accelerations.
    3) computes new forces based on updated positions.
    4) updates next step velocities using average of current and new accelerations.
    """

    def step(self, system, forces, dt, force_fn):
        """
        Advance the system by one time step using Velocity Verlet integration

        Parameters:
        -----------
        system : ParticleSystem, system to update
        forces : np.ndarray, shape (N,3)
            Forces acting on particles at current time step
        dt : float, time step
        force_fn : callable 
            Function to recompute forces from given positions
        """
        
        masses = system.masses[:, None]
        accelerations = forces / masses

        # 1) Update positions
        system.positions += system.velocities * dt + 0.5 * accelerations * dt**2

        # 2) Recompute new forces based on new positions
        new_forces = force_fn(system.positions)

        # 3) Recompute new accelerations based on new forces and new positions
        new_accelerations = new_forces / masses

        # 4) Update velocities (using average of current and new accelerations)
        system.velocities += 0.5 * (accelerations + new_accelerations) * dt

        return new_forces