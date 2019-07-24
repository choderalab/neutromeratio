import torchani
import torch
import numpy as np
from .config import nm_to_angstroms, hartree_to_kJ_mol
from simtk import unit

def ANI1ccx_force(x:np.ndarray, device, model, species, platform)->float:
    """
    Parameters
    ----------
    x : numpy array, unit'd
        coordinates
    device
    model
    species
    platform

    Returns
    -------
    F : numpy array, unit'd
        force, with units of kJ/mol/nm attached
    """

    assert(type(x) == unit.Quantity)


    # convert from nm to angstroms
    coordinates_in_angstroms = (x.in_units_of(unit.angstrom)) / unit.angstrom 
    coordinates = torch.tensor([coordinates_in_angstroms],
                               requires_grad=True, device=device, dtype=torch.float32)

    _, energy_in_hartree = model((species, coordinates))

    # convert energy from hartrees to kJ/mol
    energy_in_kJ_mol = energy_in_hartree * hartree_to_kJ_mol

    # derivative of E (in kJ/mol) w.r.t. coordinates (in nm)
    derivative = torch.autograd.grad((energy_in_kJ_mol).sum(), coordinates)[0]

    if platform == 'cpu':
        F_in_openmm_unit = - np.array(derivative)[0]
    elif platform == 'cuda':
        F_in_openmm_unit = - np.array(derivative.cpu())[0]
    else:
        raise RuntimeError('Platform needs to be specified. Either CPU or CUDA.')

    return F_in_openmm_unit * (unit.kilojoule_per_mole / unit.nanometer)


def ANI1ccx_energy(x:np.ndarray, device, model, species)->float:
    """
    Parameters
    ----------
    x : numpy array, unit'd
        coordinates
    device
    model
    species
    platform

    Returns
    -------
    E : energy in kJ/mol, unit'd
    """

    assert(type(x) == unit.Quantity)


    # convert from nm to angstroms
    coordinates_in_angstroms = (x.in_units_of(unit.angstrom)) / unit.angstrom 
    coordinates = torch.tensor([coordinates_in_angstroms],
                               requires_grad=True, device=device, dtype=torch.float32)

    _, energy_in_hartree = model((species, coordinates))

    # convert energy from hartrees to kJ/mol
    energy_in_kJ_mol = energy_in_hartree * hartree_to_kJ_mol

    return energy_in_kJ_mol.item() * unit.kilojoule_per_mole

def ANI1ccx_force_and_energy(x:np.ndarray, device, model, species, platform) -> tuple(float, float):
    """
    Parameters
    ----------
    x : numpy array, unit'd
        coordinates
    device
    model
    species
    platform

    Returns
    -------
    F : numpy array, unit'd
        force, with units of kJ/mol/nm attached
    E : energy in kJ/mol, unit'd
    """

    assert(type(x) == unit.Quantity)


    # convert from nm to angstroms
    coordinates_in_angstroms = (x.in_units_of(unit.angstrom)) / unit.angstrom 
    coordinates = torch.tensor([coordinates_in_angstroms],
                               requires_grad=True, device=device, dtype=torch.float32)

    _, energy_in_hartree = model((species, coordinates))

    # convert energy from hartrees to kJ/mol
    energy_in_kJ_mol = energy_in_hartree * hartree_to_kJ_mol

    # derivative of E (in kJ/mol) w.r.t. coordinates (in nm)
    derivative = torch.autograd.grad((energy_in_kJ_mol).sum(), coordinates)[0]

    if platform == 'cpu':
        F_in_openmm_unit = - np.array(derivative)[0]
    elif platform == 'cuda':
        F_in_openmm_unit = - np.array(derivative.cpu())[0]
    else:
        raise RuntimeError('Platform needs to be specified. Either CPU or CUDA.')

    return F_in_openmm_unit * (unit.kilojoule_per_mole / unit.nanometer), energy_in_kJ_mol.item() * unit.kilojoule_per_mole
