from simtk import unit
from .constants import nm_to_angstroms, bond_length_dict, temperature, device, mass_dict_in_daltons
import numpy as np
import torch
import logging
from scipy.stats import norm
from torch.distributions.normal import Normal
from openmmtools.constants import kB
import torch

logger = logging.getLogger(__name__)


class BaseRestraint(object):

    def __init__(self, sigma:unit.Quantity, active_at_lambda:int):
        """
        Defines a restraint base class
        Parameters
        ----------
        sigma : in angstrom
        active_at_lambda : int
            Integer to indicccate at which state the restraint is fully active. Either 0 (for 
            lambda 0), or 1 (for lambda 1) or -1 (always active)
        """

        assert(type(sigma) == unit.Quantity)
        k = (kB * temperature) / (sigma**2)
        self.device = device
        self.active_at_lambda = active_at_lambda
        self.k = torch.tensor(k.value_in_unit((unit.kilo * unit.joule) / ((unit.angstrom **2) * unit.mole)), dtype=torch.double, device=self.device, requires_grad=True)

class PointAtomRestraint(BaseRestraint):

    def __init__(self, sigma:unit.Quantity, point:np.array, active_at_lambda:int):
        """
        Defines a Point to Atom restraint base class. 

        Parameters
        ----------
        sigma : in angstrom
        point : np.array 3D, value in angstrom
        active_at_lambda : int
            Integer to indicccate at which state the restraint is fully active. Either 0 (for 
            lambda 0), or 1 (for lambda 1) or -1 (always active)
        Returns
        -------
        e : float
            bias
        """

        super().__init__(sigma, active_at_lambda)
        assert(type(point) == np.ndarray)
        self.point = torch.tensor(point, dtype=torch.double, device=self.device, requires_grad=True) 


class AtomAtomRestraint(BaseRestraint):

    def __init__(self, sigma:unit.Quantity, atom_i_idx:int, atom_j_idx:int, atoms:str, active_at_lambda:int):
        """
        Defines a Atom to Atom restraint base class. 

        Parameters
        ----------
        sigma : in angstrom
        atom_i_idx : int
            Atom i to restraint
        atom_j_idx : int
            Atom j to restraint
        atoms: str
            Str of atoms to retrieve element information
        active_at_lambda : int
            Integer to indicccate at which state the restraint is fully active. Either 0 (for 
            lambda 0), or 1 (for lambda 1) or -1 (always active)
        Returns
        -------
        e : float
            bias
        """
        super().__init__(sigma, active_at_lambda)
        self.atom_i_element = atoms[atom_i_idx]
        self.atom_j_element = atoms[atom_j_idx]
        self.atom_i_idx = atom_i_idx
        self.atom_j_idx = atom_j_idx
        try:
            self.mean_bond_length = (bond_length_dict[frozenset([self.atom_i_element, self.atom_j_element])]).value_in_unit(unit.angstrom)
        except KeyError:
            logger.critical('Bond between: {} - {}'.format(self.atom_i_element, self.atom_j_element))
            raise KeyError('Element not implemented.')
        self.upper_bound = self.mean_bond_length + 0.2
        self.lower_bound = self.mean_bond_length - 0.2


class FlatBottomRestraint(AtomAtomRestraint):

    def __init__(self, sigma:unit.Quantity, atom_i_idx:int, atom_j_idx:int, atoms:str, active_at_lambda:int=-1):
        super().__init__(sigma, atom_i_idx, atom_j_idx, atoms, active_at_lambda)

    def restraint(self, x):
        assert(type(x) == torch.Tensor)
        # x in angstrom
        distance = torch.norm(x[0][self.atom_i_idx] - x[0][self.atom_j_idx])
        if distance <= self.lower_bound:
            e = (self.k/2) * (self.lower_bound - distance.double())**2
        elif distance >= self.upper_bound:
            e = (self.k/2) * (distance.double() - self.upper_bound)**2 
        else:
            e = torch.tensor(0.0, dtype=torch.double, device=self.device)
        logging.debug('Flat bottom bias introduced: {:0.4f}'.format(e.item()))
        return e.to(device=self.device)
    

class HarmonicRestraint(AtomAtomRestraint):

    def __init__(self, sigma:unit.Quantity, atom_i_idx:int, atom_j_idx:int, atoms:str, active_at_lambda:int=-1):
        super().__init__(sigma, atom_i_idx, atom_j_idx, atoms, active_at_lambda)

    def restraint(self, x):
        assert(type(x) == torch.Tensor)
        # x in angstrom
        distance = torch.norm(x[0][self.atom_i_idx] - x[0][self.atom_j_idx]) 
        e = (self.k/2) *(distance.double() - self.mean_bond_length)**2
        logging.debug('Harmonic bias introduced: {:0.4f}'.format(e.item()))
        return e.to(device=self.device)


class FlatBottomRestraintToCenter(PointAtomRestraint):
    def __init__(self, sigma:unit.Quantity, point:unit.Quantity, radius:unit.Quantity, atom_idx:int, active_at_lambda:int=-1):
        """
        Flat well restraint that becomes active when atom moves outside of radius.
        Parameters
        ----------
        sigma : float, unit'd
        point : np.array, unit'd
        radius : float, unit'd
        atom_idx : list
            list of atoms idxs
        active_at_lambda : int
            Integer to indicccate at which state the restraint is fully active. Either 0 (for 
            lambda 0), or 1 (for lambda 1) or -1 (always active)
        """
        
        assert(type(sigma) == unit.Quantity)
        assert(type(point) == unit.Quantity)
        super().__init__(sigma, point.value_in_unit(unit.angstrom), active_at_lambda)

        self.atom_idx = atom_idx
        self.cutoff_radius = radius.value_in_unit(unit.angstrom) - 0.9 # effective radius is smaller to keep the density correct

    def restraint(self, x):
        # x in angstrom
        assert(type(x) == torch.Tensor)
        distance = torch.norm(x[0][self.atom_idx] - self.point)
        if distance >= self.cutoff_radius:
            e = (self.k/2) * (distance.double() - self.cutoff_radius)**2 
        else:
            e = torch.tensor(0.0, dtype=torch.double, device=self.device)
        logging.debug('Flat center bottom bias introduced: {:0.4f}'.format(e.item()))
        return e.to(device=self.device)

class CenterOfMassRestraint(PointAtomRestraint):

    def __init__(self, sigma:unit.Quantity, point:unit.Quantity, atom_idx:list, atoms:str, active_at_lambda:int=-1):
        """
        Center of mass restraint.

        Parameters
        ----------
        sigma : in angstrom
        point : np.array, unit'd
        atom_idx : list
            list of atoms idxs
        atoms: str
            Str of atoms to retrieve element information
        """
        assert(type(sigma) == unit.Quantity)
        assert(type(point) == unit.Quantity)
        super().__init__(sigma, point.value_in_unit(unit.angstrom), active_at_lambda)       
        self.atom_idx = atom_idx
        logger.info('Center Of Mass restraint added.')

        self.mass_list = []
        for i in atom_idx:
            self.mass_list.append(mass_dict_in_daltons[atoms[i]])
        masses = np.array(self.mass_list)
        scaled_masses = masses / masses.sum() 
        self.masses = torch.tensor(scaled_masses, dtype=torch.double, device=self.device, requires_grad=True) 

    def _calculate_center_of_mass(self, x):
        """
        Calculates the center of mass.
        One assumption that we are making here is that the ligand is at the beginning of the 
        atom str and coordinate file.
        """
        ligand_x = x[0][:len(self.mass_list)].double() # select only the ligand coordinates
        return torch.matmul(ligand_x.T, self.masses)

    def restraint(self, x):
        # x in angstrom
        assert(type(x) == torch.Tensor)

        com = self._calculate_center_of_mass(x)
        com_distance_to_point = torch.norm(com - self.point)
        e = (self.k/2) * (com_distance_to_point.sum() **2)
        return e.to(device=self.device)
