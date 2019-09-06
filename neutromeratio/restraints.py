from simtk import unit
from .constants import nm_to_angstroms
import numpy as np
import torch
from .constants import bond_length_dict
import logging
from scipy.stats import norm
from torch.distributions.normal import Normal
from openmmtools.constants import kB

logger = logging.getLogger(__name__)


def flat_bottom_position_restraint(x, tautomer_transformation:dict, atom_list:list, restrain_acceptor:bool, restrain_donor:bool, device:torch.device):
    
    """
    Applies a flat bottom positional restraint.

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

    k = 5
    if restrain_acceptor:
        if 'acceptor_hydrogen_idx' not in tautomer_transformation:
            heavy_atom_idx = tautomer_transformation['acceptor_idx']
            hydrogen_idx = tautomer_transformation['hydrogen_idx']
        else:
            heavy_atom_idx = tautomer_transformation['acceptor_idx']
            hydrogen_idx = tautomer_transformation['acceptor_hydrogen_idx']
    elif restrain_donor:
        if 'donor_hydrogen_idx' not in tautomer_transformation:
            hydrogen_idx = tautomer_transformation['hydrogen_idx']
            heavy_atom_idx = tautomer_transformation['donor_idx']
        else:
            hydrogen_idx = tautomer_transformation['donor_hydrogen_idx']
            heavy_atom_idx = tautomer_transformation['donor_idx']
    else:
        raise RuntimeError('Something went wrong.')
    
    heavy_atom_element = atom_list[heavy_atom_idx]
    mean_bond_length = bond_length_dict['{}H'.format(heavy_atom_element)]

    upper_bound = mean_bond_length.value_in_unit(unit.angstrom) + 0.2
    lower_bound = mean_bond_length.value_in_unit(unit.angstrom) - 0.2


    distance = torch.norm(x[0][hydrogen_idx] - x[0][heavy_atom_idx]) * nm_to_angstroms
    if distance <= lower_bound:
        e = k * (lower_bound - distance.double())**2
    elif distance >= upper_bound:
        e = k * (distance.double() - upper_bound)**2
    else:
        e = torch.tensor(0.0, dtype=torch.double, device=device)
    logging.debug('Flat bottom bias introduced: {:0.4f}'.format(e.item()))
    return e.to(device=device)


def harmonic_position_restraint(x, tautomer_transformation:dict, atom_list:list, restrain_acceptor:bool, restrain_donor:bool, device:torch.device):
    
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

    if restrain_acceptor:
        if 'acceptor_hydrogen_idx' not in tautomer_transformation:
            heavy_atom_idx = tautomer_transformation['acceptor_idx']
            hydrogen_idx = tautomer_transformation['hydrogen_idx']
        else:
            heavy_atom_idx = tautomer_transformation['acceptor_idx']
            hydrogen_idx = tautomer_transformation['acceptor_hydrogen_idx']
    elif restrain_donor:
        if 'donor_hydrogen_idx' not in tautomer_transformation:
            heavy_atom_idx = tautomer_transformation['donor_idx']
            hydrogen_idx = tautomer_transformation['hydrogen_idx']
        else:
            heavy_atom_idx = tautomer_transformation['donor_idx']
            hydrogen_idx = tautomer_transformation['donor_hydrogen_idx']
    else:
        raise RuntimeError('Something went wrong.')
    
    sigma = 0.1 * unit.angstrom
    T = 300 * unit.kelvin
    k = (kB * T) / (sigma**2) 
    heavy_atom_element = atom_list[heavy_atom_idx]
    mean_bond_length = (bond_length_dict['{}H'.format(heavy_atom_element)]).value_in_unit(unit.angstrom)
    logging.debug('Mean bond length: {}'.format(mean_bond_length))

    distance = torch.norm(x[0][hydrogen_idx] - x[0][heavy_atom_idx]) * nm_to_angstroms
    logging.debug('Distance: {}'.format(distance))
    k = k.value_in_unit((unit.kilo * unit.joule) / ((unit.angstrom **2) * unit.mole))
    e = (k/2) *(distance.double() - mean_bond_length)**2
    logging.debug('Harmonic bias introduced: {:0.4f}'.format(e.item()))
    return e.to(device=device)