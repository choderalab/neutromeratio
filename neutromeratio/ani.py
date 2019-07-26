import torchani
import torch
import numpy as np
from .config import nm_to_angstroms, hartree_to_kJ_mol
from simtk import unit
import simtk
from rdkit import Chem


class ANI1cxx_force_and_energy(object):

    def __init__(self, device:torch.device, model:torchani.models.ANI1ccx, species:torch.Tensor, platform:str):
        self.device = device
        self.model = model
        self.species = species
        self.platform = platform

    def calculate_force(self, x:simtk.unit.quantity.Quantity)->simtk.unit.quantity.Quantity:
        """
        Given a coordinate set the forces with respect to the coordinates are calculated.
        
        Parameters
        ----------
        x : array of floats, unit'd (distance unit)
            initial configuration

        Returns
        -------
        F : float, unit'd
            
        """

        assert(type(x) == unit.Quantity)
        coordinates = torch.tensor([x.value_in_unit(unit.nanometer)],
                                requires_grad=True, device=self.device, dtype=torch.float32)

        _, energy_in_hartree = self.model((self.species, coordinates * nm_to_angstroms))

        # convert energy from hartrees to kJ/mol
        energy_in_kJ_mol = energy_in_hartree * hartree_to_kJ_mol

        # derivative of E (in kJ/mol) w.r.t. coordinates (in nm)
        derivative = torch.autograd.grad((energy_in_kJ_mol).sum(), coordinates)[0]

        if self.platform == 'cpu':
            F = - np.array(derivative)[0]
        elif self.platform == 'cuda':
            F = - np.array(derivative.cpu())[0]
        else:
            raise RuntimeError('Platform needs to be specified. Either CPU or CUDA.')

        return F * (unit.kilojoule_per_mole / unit.nanometer)

    
    def calculate_energy(self, x:simtk.unit.quantity.Quantity)->simtk.unit.quantity.Quantity:
        """
        Given a coordinate set (x) the energy is calculated in kJ/mol.

        Parameters
        ----------
        x : array of floats, unit'd (distance unit)
            initial configuration

        Returns
        -------
        E : float, unit'd 
        """

        assert(type(x) == unit.Quantity)

        coordinates = torch.tensor([x.value_in_unit(unit.angstrom)],
                                requires_grad=True, device=self.device, dtype=torch.float32)

        _, energy_in_hartree = self.model((self.species, coordinates))

        # convert energy from hartrees to kJ/mol
        energy_in_kJ_mol = energy_in_hartree * hartree_to_kJ_mol

        return energy_in_kJ_mol.item() * unit.kilojoule_per_mole


def from_mol_to_ani_input(mol: Chem.Mol) -> dict:
    """
    Generates atom_list and coord_list entries from rdkit mol.
    Parameters
    ----------
    mol : rdkit.Chem.Mol

    Returns
    -------
    { 'atom_list' : atom_list, 'coord_list' : coord_list} 
    """
    
    atom_list = []
    coord_list = []
    for a in mol.GetAtoms():
        atom_list.append(a.GetSymbol())
        pos = mol.GetConformer().GetAtomPosition(a.GetIdx())
        coord_list.append([pos.x, pos.y, pos.z])
    return { 'atom_list' : ''.join(atom_list), 'coord_list' : coord_list}
