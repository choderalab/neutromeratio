from simtk import unit
from .constants import nm_to_angstroms
import numpy as np
import torch
from .constants import bond_length_dict
import logging

logger = logging.getLogger(__name__)


def flat_bottom_position_restraint(x, tautomer_transformation:dict, atom_list:list, restrain_acceptor:bool, restrain_donor:bool):
    
    """
    Applies a flat bottom positional restraint that mimiks a bond between .

    Parameters
    ----------
    x0 : array of floats, unit'd (distance unit)
        initial configuration
    tautomer_transformation : dict
        dictionary with index of acceptor, donor and hydrogen idx
    atom_list : list 
        list of elements
    restrain_acceptor_or_donor : str
        either 'acceptor' or 'donor'

    Returns
    -------
    e : float
        bias
    """

    k = 100
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
    if distance <= lower_bound:
        e = k * (lower_bound - distance)**2
    elif distance >= upper_bound:
        e = k * (distance - upper_bound)**2
    else:
        e = torch.tensor([0.0])
    logging.debug('Bias introduced: {:0.4f}'.format(e.item()))
    return e