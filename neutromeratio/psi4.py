from .constants import hartree_to_kJ_mol
from simtk import unit
import psi4
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
psi4.core.set_output_file("output1.dat", True)
psi4.set_memory('4 GB')
psi4.set_num_threads(4)


def mol2psi4(mol: Chem.Mol, conformer_id: int = 0) -> psi4.core.Molecule:
    """Returns a psi4 Molecule object instance for a single conformer from a rdkit mol object.

    Parameters
    ----------
    mol : Chem.Mol
        rdkit mol object
    conformer_id : int
        specifies the conformer to use

    Returns
    -------
    mol : psi4.core.Molecule
        a psi4 molecule object instance
    """

    assert(type(mol) == Chem.Mol)
    atoms = mol.GetAtoms()
    string = "\n"
    for _, atom in enumerate(atoms):
        pos = mol.GetConformer(conformer_id).GetAtomPosition(atom.GetIdx())
        string += "{} {} {} {}\n".format(atom.GetSymbol(), pos.x, pos.y, pos.z)
    string += "units angstrom\n"
    return psi4.geometry(string)


def optimize(mol: psi4.core.Molecule, method: str = 'wB97X/6-31g*') -> unit.Quantity:
    """Runs a minimization for a psi4 molecule object instance using a specified method (default: wB97X/6-31g*).

    Parameters
    ----------
    mol : psi4.core.Molecule
        psi4 object instance
    method : str
        specifies the method to use

    Returns
    -------
    energy : unit.Quantity
        energy of optimized geometry
    """

    e = psi4.optimize(method, molecule=mol)
    return (e * hartree_to_kJ_mol) * unit.kilojoule_per_mole


def calculate_energy(mol: str, method: str = 'wB97X/6-31g*') -> unit.Quantity:
    """Calculates the single point energy of a psi4 molecule object instance.

    Parameters
    ----------
    mol : psi4.core.Molecule
        psi4 molecule object instance
    method : str
        specifies the method to use

    Returns
    -------
    energy : unit.Quantity
        energy of optimized geometry
    """
    e = psi4.energy(method, molecule=mol) * hartree_to_kJ_mol
    return e * unit.kilojoule_per_mole
