from scipy.stats import norm
import random
import numpy as np
from simtk import unit
import copy
from .utils import reduced_pot, generate_xyz_string
import math
import logging

logger = logging.getLogger(__name__)


class Instantaneous_MC_Mover(object):
    
    def __init__(self, 
                donor_idx:int, 
                hydrogen_idx:int, 
                acceptor_idx:int, 
                atom_list:str, 
                energy_function):

        self.donor_idx = donor_idx
        self.hydrogen_idx = hydrogen_idx
        self.acceptor_idx = acceptor_idx
        self.atom_list = atom_list
        self.energy_function = energy_function
        self.mc_accept_counter = 0
        self.mc_reject_counter = 0
        
        self.bond_length_dict = {'CH' : 1.09 * unit.angstrom,
                                'OH' : 0.96 * unit.angstrom,
                                'NH' : 1.01 * unit.angstrom
                                 }

        # element of the hydrogen acceptor and donor
        self.acceptor_element = self.atom_list[self.acceptor_idx]
        self.donor_element = self.atom_list[self.donor_idx]
        # the equilibrium bond length taken from self.bond_length_dict
        self.acceptor_hydrogen_equilibrium_bond_length = self.bond_length_dict['{}H'.format(self.acceptor_element)]
        self.donor_hydrogen_equilibrium_bond_length = self.bond_length_dict['{}H'.format(self.donor_element)]
        # the stddev for the bond length
        self.acceptor_hydrogen_stddev_bond_length = 0.15 * unit.angstrom
        self.donor_hydrogen_stddev_bond_length = 0.15 * unit.angstrom
        # the mean bond length is the bond length that is actually used for proposing coordinates
        self.proposed_coordinates = []
        self.initial_coordinates = []
        self.work_values = []

    def write_xyz_files(self, ts:int, filename:str):
        """
        Writes xyz files in current directory.
        The files are saved in {name}_ts{ts}_initial.xyz and {name}_ts{ts}_proposed.xyz.
        Parameters
        ----------
        atoms: list of atoms (in a single string) 
        coordinates: numpy array with coordinates
        name: name of the file
        """

        coordinates = self.initial_coordinates[ts]
        f_initial = open(f"{filename}_ts{ts}_initial.xyz", 'w')
        xyz_string = generate_xyz_string(self.atom_list, coordinates)
        for line in xyz_string:
            f_initial.write(line)
        f_initial.close()

        coordinates = self.proposed_coordinates[ts]
        f_proposed = open(f"{filename}_ts{ts}_proposed.xyz", 'w')
        xyz_string = generate_xyz_string(self.atom_list, coordinates)
        for line in xyz_string:
            f_proposed.write(line)
        f_proposed.close()


    def perform_mc_move(self, coordinates:unit.quantity.Quantity):
        """
        Moves a hydrogen (self.hydrogen_idx) from a starting position connected to a heavy atom
        donor (self.donor_idx) to a new position around an acceptor atom (self.acceptor_idx).
        The new coordinates are sampled from a radial distribution, with the center being the acceptor atom,
        the mean: mean_bond_length = self.acceptor_hydrogen_equilibrium_bond_length * self.acceptor_mod_bond_length
        and standard deviation: self.acceptor_hydrogen_stddev_bond_length.
        Calculates the log probability of the forward and reverse proposal and returns the work.
        """
        # convert coordinates to angstroms
        # coordinates_before_move
        coordinates_A = coordinates.in_units_of(unit.angstrom)
        #coordinates_after_move
        coordinates_B = self._move_hydrogen_to_acceptor_idx(copy.deepcopy(coordinates_A))
        energy_B = self.energy_function.calculate_energy(coordinates_B)
        energy_A = self.energy_function.calculate_energy(coordinates_A)
        delta_u = reduced_pot(energy_B - energy_A)

        # log probability of forward proposal from the initial coordinates (coordinate_A) to proposed coordinates (coordinate_B)
        log_p_forward = self.log_probability_of_proposal_to_B(coordinates_A, coordinates_B)
        # log probability of reverse proposal given the proposed coordinates (coordinate_B)
        log_p_reverse = self.log_probability_of_proposal_to_A(coordinates_B, coordinates_A)
        work = - ((- delta_u) + (log_p_reverse - log_p_forward))
        return coordinates_B, work

    def _log_probability_of_radial_proposal(self, r:float, r_mean:float, r_stddev:float) -> float:
        """Logpdf of N(r : mu=r_mean, sigma=r_stddev)"""
        return norm.logpdf(r, loc=r_mean, scale=r_stddev)

    def log_probability_of_proposal_to_B(self, X:unit.quantity.Quantity, X_prime:unit.quantity.Quantity) -> float:
        """log g_{A \to B}(X --> X_prime)"""
        # test if acceptor atoms are similar in both coordinate sets
        assert(np.allclose(X[self.acceptor_idx], X_prime[self.acceptor_idx]))
        # calculates the effective bond length of the proposed conformation between the
        # hydrogen atom and the acceptor atom
        r = np.linalg.norm(X_prime[self.hydrogen_idx] - X_prime[self.acceptor_idx])
        return self._log_probability_of_radial_proposal(r, self.acceptor_hydrogen_equilibrium_bond_length * (1/unit.angstrom), self.acceptor_hydrogen_stddev_bond_length * (1/unit.angstrom))

    def log_probability_of_proposal_to_A(self, X:unit.quantity.Quantity, X_prime:unit.quantity.Quantity) -> float:
        """log g_{B \to A}(X --> X_prime)"""
        assert(np.allclose(X[self.donor_idx], X_prime[self.donor_idx]))
        # calculates the effective bond length of the initial conformation between the
        # hydrogen atom and the donor atom
        r = np.linalg.norm(X_prime[self.hydrogen_idx] - X_prime[self.donor_idx])
        return self._log_probability_of_radial_proposal(r, self.donor_hydrogen_equilibrium_bond_length * (1/unit.angstrom), self.donor_hydrogen_stddev_bond_length * (1/unit.angstrom))

    def accept_reject(self, log_P_accept: float) -> bool:
        """Perform acceptance/rejection check according to the Metropolis-Hastings acceptance criterion."""
        # think more about symmetric
        return (log_P_accept > 0.0) or (random.random() < math.exp(log_P_accept))

    def _move_hydrogen_to_acceptor_idx(self, coordinates:unit.quantity.Quantity) -> unit.quantity.Quantity:
        """Moves a single hydrogen (specified in self.hydrogen_idx) from a donor
        atom (self.donor_idx) to a new position around the acceptor atom (self.acceptor_idx).
        Parameters
        ----------
        coordinates_in_angstroms :numpy array, unit'd
            coordinates

        Returns
        -------
        coordinates_in_angstroms :numpy array, unit'd
            coordinates
        """
        def sample_spherical(ndim=3):
            """
            Generates new coordinates for a hydrogen around a heavy atom acceptor.
            Bond length is defined by the hydrogen - acceptor element equilibrium bond length,
            defined in self.bond_length_dict. Standard deviation of bond length is defined
            in self.std_bond_length.
            A bias to the bond length can be introduced through self.mod_bond_length.
            The effective bond length is mean_bond_length *= self.mod_bond_length.
            """
            # sample a random direction
            unit_vector = np.random.randn(ndim)
            unit_vector /= np.linalg.norm(unit_vector, axis=0)
            # sample a random length
            effective_bond_length = (np.random.randn() * self.acceptor_hydrogen_stddev_bond_length + self.acceptor_hydrogen_equilibrium_bond_length)
            return (unit_vector * effective_bond_length)


        # get coordinates of acceptor atom and hydrogen that moves
        acceptor_coordinate = coordinates[self.acceptor_idx]

        # generate new hydrogen atom position and replace old coordinates
        new_hydrogen_coordinate = acceptor_coordinate + sample_spherical()
        coordinates[self.hydrogen_idx] = new_hydrogen_coordinate

        return coordinates



class NonequilibriumMC(object):

    def __init__(self,
            donor_idx:int,
            hydrogen_idx:int,
            dummy_hydrogen_idx:int,
            acceptor_idx:int,
            atom_list:str,
            energy_function):

        self.donor_idx = donor_idx
        self.hydrogen_idx = hydrogen_idx
        self.dummy_hydrogen_idx = dummy_hydrogen_idx
        self.acceptor_idx = acceptor_idx
        self.atom_list = atom_list
        self.energy_function = energy_function
        self.work_values = []


# E_j(lambda) = (1 - lambda ) * E_j_without_i + lambda * E_j_with_i
