import numpy as np 

def lennard_jones_forces(positions, epsilon = 1.0, sigma = 1.0):
    """
    Compute Lennard-Jones forces for all particle pairs.

    LJ potential between particles i and j:
        V(r) = 4 * epsilon * [ (sigma / r)**12 - (sigma / r)**6 ]
    
    The force is the negative gradient of the potential:
        F(r) = -dV/dr
    which at sigma = r_ij gives:
        F(r) = 24 * epsilon * [ 2*(r**-13) - (r**-7) ] * r_ij 


    Parameters
    -----------
    :param positions: np.darray, shape (N,3)
    :param epsilon: float
    :param sigma: float

    Returns
    -----------
    :return forces : np.ndarray, shape (N,3 )
    """

    n = positions.shape[0]
    forces = np.zeros_like(positions)

    for i in range(n):
        for j in range(i+1, n):
            r_ij = positions[j] - positions[i]
            r = np.linalg.norm(r_ij)

            if r == 0:
                continue  # avoid division by zero
            # Lennard-Jones force calculation
            inv_r = sigma / r
            inv_r6 = inv_r ** 6
            inv_r12 = inv_r ** 12

            magnitude = 24 * epsilon / r * (2* inv_r12 - inv_r6)
            force_vec = magnitude * (r_ij / r)

            forces[i] -= force_vec
            forces[j] += force_vec  #Newton's 3rd law
    
    return forces
            
