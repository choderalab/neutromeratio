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


class Restraint(object):

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

        assert(type(sigma) == unit.Quantity)
        k = (kB * temperature) / (sigma**2)
        self.k = k.value_in_unit((unit.kilo * unit.joule) / ((unit.angstrom **2) * unit.mole))
        self.device = device
        self.atom_i_element = atoms[atom_i_idx]
        self.atom_j_element = atoms[atom_j_idx]
        self.atom_i_idx = atom_i_idx
        self.atom_j_idx = atom_j_idx
        self.mean_bond_length = (bond_length_dict[frozenset([self.atom_i_element, self.atom_j_element])]).value_in_unit(unit.angstrom)
        self.active_at_lambda = active_at_lambda
        self.upper_bound = self.mean_bond_length + 0.2
        self.lower_bound = self.mean_bond_length - 0.2


    def flat_bottom_position_restraint(self, x):

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
    
    
    def harmonic_position_restraint(self, x):

        # x in angstrom
        distance = torch.norm(x[0][self.atom_i_idx] - x[0][self.atom_j_idx]) 
        e = (self.k/2) *(distance.double() - self.mean_bond_length)**2
        logging.debug('Harmonic bias introduced: {:0.4f}'.format(e.item()))
        return e.to(device=self.device)









# def flat_bottom_position_restraint(x, tautomer_transformation:dict, atom_list:list, restrain_acceptor:bool, restrain_donor:bool, device:torch.device):
    
#     """
#     Applies a flat bottom positional restraint.

#     Parameters
#     ----------
#     x0 : array of floats, unit'd (distance unit)
#         initial configuration
#     tautomer_transformation : dict
#         dictionary with index of acceptor, donor and hydrogen idx
#     atom_list : list 
#         list of elements
#     restrain_acceptor : bool
#         should the acceptor be restraint
#     restrain_donor : boold
#         should the donor be restraint

#     Returns
#     -------
#     e : float
#         bias
#     """

#     sigma = 0.1 * unit.angstrom
#     T = 300 * unit.kelvin
#     k = (kB * T) / (sigma**2)

#     if restrain_acceptor:
#         heavy_atom_idx = tautomer_transformation['acceptor_idx']
#         if 'acceptor_hydrogen_idx' in tautomer_transformation:
#             hydrogen_idx = tautomer_transformation['acceptor_hydrogen_idx']
#         else:
#             hydrogen_idx = tautomer_transformation['hydrogen_idx']
#     elif restrain_donor:
#         heavy_atom_idx = tautomer_transformation['donor_idx']
#         if 'donor_hydrogen_idx' in tautomer_transformation:
#             hydrogen_idx = tautomer_transformation['donor_hydrogen_idx']
#         else:
#             hydrogen_idx = tautomer_transformation['hydrogen_idx']
#     else:
#         raise RuntimeError('Something went wrong.')
    
#     heavy_atom_element = atom_list[heavy_atom_idx]
#     mean_bond_length = bond_length_dict['{}H'.format(heavy_atom_element)]

#     upper_bound = mean_bond_length.value_in_unit(unit.angstrom) + 0.2
#     lower_bound = mean_bond_length.value_in_unit(unit.angstrom) - 0.2

#     k = k.value_in_unit((unit.kilo * unit.joule) / ((unit.angstrom **2) * unit.mole))
#     distance = torch.norm(x[0][hydrogen_idx] - x[0][heavy_atom_idx]) * nm_to_angstroms
#     if distance <= lower_bound:
#         e = (k/2) * (lower_bound - distance.double())**2
#     elif distance >= upper_bound:
#         e = (k/2) * (distance.double() - upper_bound)**2 
#     else:
#         e = torch.tensor(0.0, dtype=torch.double, device=device)
#     logging.debug('Flat bottom bias introduced: {:0.4f}'.format(e.item()))
#     return e.to(device=device)


# def harmonic_position_restraint(x, tautomer_transformation:dict, atom_list:list, restrain_acceptor:bool, restrain_donor:bool, device:torch.device):
    
#     """
#     Applies a gaussian positional restraint.

#     Parameters
#     ----------
#     x0 : array of floats, unit'd (distance unit)
#         initial configuration
#     tautomer_transformation : dict
#         dictionary with index of acceptor, donor and hydrogen idx
#     atom_list : list 
#         list of elements
#     restrain_acceptor : bool
#         should the acceptor be restraint
#     restrain_donor : boold
#         should the donor be restraint

#     Returns
#     -------
#     e : float
#         bias
#     """

#     if restrain_acceptor:
#         heavy_atom_idx = tautomer_transformation['acceptor_idx']
#         if 'acceptor_hydrogen_idx' in tautomer_transformation:
#             hydrogen_idx = tautomer_transformation['acceptor_hydrogen_idx']
#         else:
#             hydrogen_idx = tautomer_transformation['hydrogen_idx']
#     elif restrain_donor:
#         heavy_atom_idx = tautomer_transformation['donor_idx']
#         if 'donor_hydrogen_idx' in tautomer_transformation:
#             hydrogen_idx = tautomer_transformation['donor_hydrogen_idx']
#         else:
#             hydrogen_idx = tautomer_transformation['hydrogen_idx']
#     else:
#         raise RuntimeError('Something went wrong.')
    
#     sigma = 0.1 * unit.angstrom
#     T = 300 * unit.kelvin
#     k = (kB * T) / (sigma**2) 
#     heavy_atom_element = atom_list[heavy_atom_idx]
#     mean_bond_length = (bond_length_dict['{}H'.format(heavy_atom_element)]).value_in_unit(unit.angstrom)
#     logging.debug('Mean bond length: {}'.format(mean_bond_length))

#     distance = torch.norm(x[0][hydrogen_idx] - x[0][heavy_atom_idx]) * nm_to_angstroms
#     logging.debug('Distance: {}'.format(distance))
#     k = k.value_in_unit((unit.kilo * unit.joule) / ((unit.angstrom **2) * unit.mole))
#     e = (k/2) *(distance.double() - mean_bond_length)**2
#     logging.debug('Harmonic bias introduced: {:0.4f}'.format(e.item()))
#     return e.to(device=device)