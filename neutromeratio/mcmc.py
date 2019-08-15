from scipy.stats import norm
import random
import numpy as np
from simtk import unit
import copy
from .utils import reduced_pot
import math
import logging
from tqdm import tqdm
from .equilibrium import LangevinDynamics
from .constants import bond_length_dict

logger = logging.getLogger(__name__)


class MC_Mover(object):

    def __init__(self, 
                donor_idx:int, 
                hydrogen_idx:int, 
                acceptor_idx:int, 
                atom_list:str, 
                energy_function,
                langevin_dynamics:LangevinDynamics):

        self.donor_idx = donor_idx
        self.hydrogen_idx = hydrogen_idx
        self.acceptor_idx = acceptor_idx
        self.atom_list = atom_list
        self.energy_function = energy_function
        self.langevin_dynamics = langevin_dynamics
        self.mc_accept_counter = 0
        self.mc_reject_counter = 0
        
        self.bond_length_dict = bond_length_dict
        # element of the hydrogen acceptor and donor
        self.acceptor_element = self.atom_list[self.acceptor_idx]
        self.donor_element = self.atom_list[self.donor_idx]
        # the equilibrium bond length taken from self.bond_length_dict
        self.acceptor_hydrogen_equilibrium_bond_length = self.bond_length_dict['{}H'.format(self.acceptor_element)]
        self.donor_hydrogen_equilibrium_bond_length = self.bond_length_dict['{}H'.format(self.donor_element)]
        # the stddev for the bond length
        self.acceptor_hydrogen_stddev_bond_length = 0.15 * unit.angstrom
        self.donor_hydrogen_stddev_bond_length = 0.15 * unit.angstrom
        self.proposed_coordinates = []
        self.initial_coordinates = []
        self.work_values = []

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



class Instantaneous_MC_Mover(MC_Mover):


    def perform_md_mc_protocol(self,
                               x0:unit.quantity.Quantity,
                               nr_of_mc_trials:int = 500,
                               nr_of_md_steps:int = 100,
                               ):
        """
        Performing instantaneous MC and langevin dynamics.
        Given a coordinate set the forces with respect to the coordinates are calculated.
        
        Parameters
        ----------
        x0 : array of floats, unit'd (distance unit)
            initial configuration
        nr_of_mc_trials:int
                        nr of MC moves that should be performed


        Returns
        -------
        traj : array of floats, unit'd (distance unit)
        """    

        trange = tqdm(range(nr_of_mc_trials))
        traj_in_nm = []
        work_values = []

        for _ in trange:

            trajectory = self.langevin_dynamics.run_dynamics(x0, nr_of_md_steps)
            final_coordinate_set = trajectory[-1]
            traj_in_nm += [x / unit.nanometer for x in trajectory]      
            # MC move
            new_coordinates, work = self.perform_mc_move(final_coordinate_set)
            self.proposed_coordinates.append(new_coordinates)
            self.initial_coordinates.append(final_coordinate_set)
            work_values.append(work)
            # update new coordinates for langevin dynamics
            x0 = final_coordinate_set

        return work_values, traj_in_nm



class NonequilibriumMC(MC_Mover):

    def perform_md_mc_protocol(self,
                               x0:unit.quantity.Quantity,
                               nr_of_mc_trials:int = 500,
                               nr_of_md_steps:int = 20
                               ):
        """
        """    

        traj_in_nm = []       
        work_values = []

        # initial sampling
        trajectory = self.langevin_dynamics.run_dynamics(x0, 5000)
        x0 = trajectory[-1]


        logging.info('Decoupling hydrogen ...')
        # decouple the hydrogen from the environment
        self.energy_function.restrain_donor = True
        self.energy_function.restrain_acceptor = False


        for lambda_value in tqdm(np.linspace(1, 0, nr_of_mc_trials/2)):
            
            trajectory = self.langevin_dynamics.run_dynamics(x0, nr_of_md_steps)
            final_coordinate_set = trajectory[-1]
            work_values.append(self.pertubate_lambda(final_coordinate_set, lambda_value))
            # update new coordinates for langevin dynamics
            x0 = final_coordinate_set
            traj_in_nm += [x / unit.nanometer for x in trajectory]

        logging.info('Moving hydrogen ...')
        # turn of the bond restraint
        # move the hydrogen to a new position
        x0, work_of_hydrogen_move = self.perform_mc_move(copy.deepcopy(final_coordinate_set))
        traj_in_nm += [x0 / unit.nanometer]

        work_values.append(work_of_hydrogen_move)
        logging.info('Work of Hydrogen move: {:0.4f}'.format(work_of_hydrogen_move))
        logging.info('Recoupling hydrogen ...')
        # couple the hydrogen to new environment
        # turn on the bond restraint
        self.energy_function.restrain_donor = False
        self.energy_function.restrain_acceptor = True
        for lambda_value in tqdm(np.linspace(0, 1, nr_of_mc_trials/2)):
            
            trajectory = self.langevin_dynamics.run_dynamics(x0, nr_of_md_steps)
            final_coordinate_set = trajectory[-1]
            work_values.append(self.pertubate_lambda(final_coordinate_set, lambda_value))
            # update new coordinates for langevin dynamics
            x0 = final_coordinate_set
            traj_in_nm += [x / unit.nanometer for x in trajectory]

       
        return work_values, traj_in_nm


    def pertubate_lambda(self, coordinates:unit.quantity.Quantity, lambda_value:float):
        """
        Lambda value that controls the coupling of the hydrogen to the environment is 
        propagated here.
        """

        coordinates = coordinates.in_units_of(unit.angstrom)
        # energy of current state
        energy_A = self.energy_function.calculate_energy(coordinates)
        # set new lambda value
        self.energy_function.lambda_value = lambda_value
        # energy of propagated state
        energy_B = self.energy_function.calculate_energy(coordinates)
        return reduced_pot(energy_B - energy_A)

    def perform_mc_move(self, coordinates:unit.quantity.Quantity):
        """
        Moves a hydrogen (self.hydrogen_idx) from a starting position connected to a heavy atom
        donor (self.donor_idx) to a new position around an acceptor atom (self.acceptor_idx).
        The new coordinates are sampled from a radial distribution, with the center being the acceptor atom,
        the mean: mean_bond_length = self.acceptor_hydrogen_equilibrium_bond_length * self.acceptor_mod_bond_length
        and standard deviation: self.acceptor_hydrogen_stddev_bond_length.
        Calculates the log probability of the forward and reverse proposal and returns the work.
        """

        # coordinates_before_move
        coordinates_A = coordinates.in_units_of(unit.angstrom)
        self.energy_function.restrain_donor = True
        self.energy_function.restrain_acceptor = False
        energy_A = self.energy_function.calculate_energy(coordinates_A)

        #coordinates_after_move
        coordinates_B = self._move_hydrogen_to_acceptor_idx(copy.deepcopy(coordinates_A))
        self.energy_function.restrain_donor = False
        self.energy_function.restrain_acceptor = True
        energy_B = self.energy_function.calculate_energy(coordinates_B)

        delta_u = reduced_pot(energy_B - energy_A)
        logging.info('delta_u : {:0.4f}'.format(delta_u))
        # log probability of forward proposal from the initial coordinates (coordinate_A) to proposed coordinates (coordinate_B)
        log_p_forward = self.log_probability_of_proposal_to_B(coordinates_A, coordinates_B)
        # log probability of reverse proposal given the proposed coordinates (coordinate_B)
        log_p_reverse = self.log_probability_of_proposal_to_A(coordinates_B, coordinates_A)
        work = - ((- delta_u) + (log_p_reverse - log_p_forward))
        return coordinates_B, work
