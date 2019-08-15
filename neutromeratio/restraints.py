from simtk import unit
from .constants import nm_to_angstroms
import numpy as np
import torch
from .constants import bond_length_dict
import logging

logger = logging.getLogger(__name__)


class Restraint():
    def __init__(self, heavy_atom_index, hydrogen_index):
        self.heavy_atom_index = heavy_atom_index
        self.hydrogen_index = hydrogen_index

    def forward(self, x):
        raise NotImplementedError




class FlatBottomRestraint(Restraint):
    def __init__(self, heavy_atom_index, hydrogen_index,
                 min_dist=0.8 * unit.angstrom,
                 max_dist=1.2 * unit.angstrom,
                 spring_constant=10):
        super().__init__(heavy_atom_index, hydrogen_index)
        self.min_dist_in_angstroms = min_dist.value_in_unit(unit.angstrom)
        self.max_dist_in_angstroms = max_dist.value_in_unit(unit.angstrom)
        self.spring_constant = spring_constant
        # TODO: units on spring_constant

    def forward(self, x):
        """Assumes x is in units of nanometers"""
        assert(len(x) == 1) # TODO: assumes x is a [1, n_atoms, 3] tensor
        distance_in_angstroms = (torch.norm(x[0][self.hydrogen_index] - x[0][self.heavy_atom_index]) * nm_to_angstroms).double()

        left_penalty = (distance_in_angstroms < self.min_dist_in_angstroms) * (self.spring_constant * (self.min_dist_in_angstroms - distance_in_angstroms) ** 2)
        right_penalty = (distance_in_angstroms > self.max_dist_in_angstroms) * (self.spring_constant * (distance_in_angstroms - self.max_dist_in_angstroms) ** 2)
        return left_penalty + right_penalty

def gaussian_position_restraint(x, tautomer_transformation:dict, atom_list:list, restrain_acceptor:bool, restrain_donor:bool):
    
    """
    Applies a gaussian positional restraint.

    Parameters
    ----------
    x0 : array of floats, unit'd (distance unit)
        initial configuration
    tautomer_transformation : dict
        dictionary with index of acceptor, donor and hydrogen idx
    atom_list : list 
        list of elements
    restrain_acceptor : bool
        should the acceptor be restraint
    restrain_donor : boold
        should the donor be restraint

    Returns
    -------
    e : float
        bias
    """

    k = 10
    if restrain_acceptor:
        heavy_atom_idx = tautomer_transformation['acceptor_idx']
    elif restrain_donor:
        heavy_atom_idx = tautomer_transformation['donor_idx']
    else:
        raise RuntimeError('Something went wrong.')
    
    heavy_atom_element = atom_list[heavy_atom_idx]
    mean_bond_length = bond_length_dict['{}H'.format(heavy_atom_element)]

    upper_bound = mean_bond_length.value_in_unit(unit.angstrom) + 0.2
    lower_bound = mean_bond_length.value_in_unit(unit.angstrom) - 0.2
    distance = torch.norm(x[0][tautomer_transformation['hydrogen_idx']] - x[0][heavy_atom_idx]) * nm_to_angstroms
    

    # PLACEHOLDER FUNCTION
    e = k * (lower_bound - distance)**2
    
    logging.debug('Bias introduced: {:0.4f}'.format(e.item()))
    return e