from minimd.potential_energy import lennard_jones_potential
from minimd.system import ParticleSystem


def total_energy(system):
    kinetic = system.kinetic_energy()
    potential = lennard_jones_potential(system.positions)
    return kinetic + potential