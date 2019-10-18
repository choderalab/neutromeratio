from simtk import unit
from .constants import nm_to_angstroms, bond_length_dict, temperature, device
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

        assert(type(sigma) == unit.Quantity)
        k = (kB * temperature) / (sigma**2)
        self.k = k.value_in_unit((unit.kilo * unit.joule) / ((unit.angstrom **2) * unit.mole))
        self.device = device
        self.active_at_lambda = active_at_lambda

class PointAtomRestraint(BaseRestraint):

    def __init__(self, sigma:unit.Quantity, point:unit.Quantity, active_at_lambda:int):
        """
        Defines a restraint. 

        Parameters
        ----------
        sigma : in angstrom
        point : np.array 3D, unit'd
        active_at_lambda : int
            Integer to indicccate at which state the restraint is fully active. Either 0 (for 
            lambda 0), or 1 (for lambda 1) or -1 (always active)
        Returns
        -------
        e : float
            bias
        """

        super().__init__(sigma, active_at_lambda)
        assert(type(point) == unit.Quantity)
        self.point = torch.from_numpy(point).to(dtype=torch.double, device=self.device) 



class AtomAtomRestraint(BaseRestraint):

    def __init__(self, sigma:unit.Quantity, atom_i_idx:int, atom_j_idx:int, atoms:str, active_at_lambda:int):
        """
        Defines a restraint. 

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
        self.mean_bond_length = (bond_length_dict[frozenset([self.atom_i_element, self.atom_j_element])]).value_in_unit(unit.angstrom)
        self.upper_bound = self.mean_bond_length + 0.2
        self.lower_bound = self.mean_bond_length - 0.2


class FlatBottomRestraint(AtomAtomRestraint):

    def __init__(self, sigma:unit.Quantity, atom_i_idx:int, atom_j_idx:int, atoms:str, active_at_lambda:int=-1):
        super().__init__(sigma, atom_i_idx, atom_j_idx, atoms, active_at_lambda)

    def restraint(self, x):

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

        # x in angstrom
        distance = torch.norm(x[0][self.atom_i_idx] - x[0][self.atom_j_idx]) 
        e = (self.k/2) *(distance.double() - self.mean_bond_length)**2
        logging.debug('Harmonic bias introduced: {:0.4f}'.format(e.item()))
        return e.to(device=self.device)


class FlatBottomRestraintToCenter(PointAtomRestraint):
    def __init__(self, sigma:unit.Quantity, point:unit.Quantity, radius:unit.Quantity, atom_idx:int, active_at_lambda:int=-1):
        """
        Flat well restraint that becomes active when water moves outside of radius.
        """
        
        assert(type(sigma) == unit.Quantity)
        super().__init__(sigma, point, active_at_lambda)

        self.atom_idx = atom_idx
        self.cutoff_radius = radius.value_in_unit(unit.angstrom) + 0.2

    def restraint(self, x):

        # x in angstrom
        distance = torch.norm(x[0][self.atom_idx] - self.point)
        if distance >= self.cutoff_radius:
            e = (self.k/2) * (distance.double() - self.cutoff_radius)**2 
        else:
            e = torch.tensor(0.0, dtype=torch.double, device=self.device)
        logging.debug('Flat center bottom bias introduced: {:0.4f}'.format(e.item()))
        return e.to(device=self.device)

class CenterOfMassRestraint(object):

    def __init__(self, sigma:unit.Quantity, radius:unit.Quantity, atom_idx:int, active_at_lambda:int=-1):
        assert(type(sigma) == unit.Quantity)
        k = (kB * temperature) / (sigma**2)
        self.k = k.value_in_unit((unit.kilo * unit.joule) / ((unit.angstrom **2) * unit.mole))
        self.device = device
        self.atom_idx = atom_idx
        self.active_at_lambda = active_at_lambda
        self.cutoff_radius = radius.value_in_unit(unit.angstrom) + 0.2
        self.radius = radius.value_in_unit(unit.angstrom)
        self.center = torch.tensor([self.radius, self.radius, self.radius], dtype=torch.double, device=self.device)

    def restraint(self, x):

        # x in angstrom
        distance = torch.norm(x[0][self.atom_idx] - self.center)
        if distance >= self.cutoff_radius:
            e = (self.k/2) * (distance.double() - self.cutoff_radius)**2 
        else:
            e = torch.tensor(0.0, dtype=torch.double, device=self.device)
        logging.debug('Flat center bottom bias introduced: {:0.4f}'.format(e.item()))
        return e.to(device=self.device)
