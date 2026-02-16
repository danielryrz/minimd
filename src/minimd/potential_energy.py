import numpy as np

def lennard_jones_potential(positions, epsilon=1.0, sigma=1.0):
    """
    Compute total Lennard-Jones potential energy for a system of particles.

    LJ potential between particles i and j:
        V(r) = 4 * epsilon * [ (sigma / r)**12 - (sigma / r)**6 ]
    
    Parameters
    -----------
    :param positions: np.darray, shape (N,3)
    :param epsilon: float
    :param sigma: float
    Returns
    -----------
    :return total_potential : float
    """

    n = positions.shape[0]
    energy = 0.0 

    for i in range(n):
        for j in range(i+1,n):
            r = np.linalg.norm(positions[j] - positions[i])
            if r == 0:
                continue  # avoid division by zero

            inv_r = sigma / r
            inv_r6 = inv_r ** 6
            inv_r12 = inv_r ** 12

            energy += 4 * epsilon * (inv_r12 - inv_r6)
    
    return energy