# Add imports here
from simtk import unit
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import mdtraj as md
import numpy as np
from .constants import kT
import nglview
import logging
from pdbfixer import PDBFixer
from simtk.openmm import Vec3
import random

logger = logging.getLogger(__name__)

def is_quantity_close(quantity1, quantity2, rtol=1e-10, atol=0.0):
    """Check if the quantities are equal up to floating-point precision errors.
    Parameters
    ----------
    quantity1 : simtk.unit.Quantity
        The first quantity to compare.
    quantity2 : simtk.unit.Quantity
        The second quantity to compare.
    rtol : float, optional
        Relative tolerance (default is 1e-10).
    atol : float, optional
        Absolute tolerance (default is 0.0).
    Returns
    -------
    True if the quantities are equal up to approximately 10 digits.
    Raises
    ------
    TypeError
        If the two quantities are of incompatible units.
    """
    if not quantity1.unit.is_compatible(quantity2.unit):
        raise TypeError('Cannot compare incompatible quantities {} and {}'.format(
            quantity1, quantity2))

    value1 = quantity1.value_in_unit_system(unit.md_unit_system)
    value2 = quantity2.value_in_unit_system(unit.md_unit_system)

    # np.isclose is not symmetric, so we make it so.
    if abs(value2) >= abs(value1):
        return np.isclose(value1, value2, rtol=rtol, atol=atol)
    else:
        return np.isclose(value2, value1, rtol=rtol, atol=atol)

  
def write_pdb(mol:Chem.Mol, filepath:str)->str:
    """
    Writes pdb file in path directory. If directory does not exist it is created.
    Parameters
    ----------
    mol: the mol that should be saved.
    filepath
    
    Returns
    ----------
    PDBfile as string
    """

    Chem.MolToPDBFile(mol, filepath)
    return Chem.MolToPDBBlock(mol)

def add_solvent(pdb_filepath:str, ani_input:dict, pdb_output_filepath:str, box_length:unit.quantity.Quantity=(2.5 * unit.nanometer)):
    
    assert(type(box_length) == unit.Quantity)

    pdb = PDBFixer(filename=pdb_filepath)
    # Step 0: put the ligand in the center
    #pdb.positions = np.array(pdb.positions.value_in_unit(unit.nanometer)) + box_length/2
    # add water
    l = box_length.value_in_unit(unit.nanometer)
    pdb.addSolvent(boxVectors=(Vec3(l, 0.0, 0.0), Vec3(0.0, l, 0.0), Vec3(0.0, 0.0, l)))
    # Step 1: convert coordinates from standard cartesian coordinate to unit
    # cell coordinates
    #inv_cell = 1/box_length
    #coordinates_cell = np.array(pdb.positions.value_in_unit(unit.nanometer)) * inv_cell
    # Step 2: wrap cell coordinates into [0, 1)
    #coordinates_cell -= np.floor(coordinates_cell)
    # Step 3: convert back to coordinates
    #coordinates_cell = (coordinates_cell * box_length) * unit.nanometer
    #pdb.positions = coordinates_cell
    from simtk.openmm.app import PDBFile
    PDBFile.writeFile(pdb.topology, pdb.positions, open(pdb_output_filepath, 'w'))
    
    atom_list = []
    coord_list = []
    for atom, coor in zip(pdb.topology.atoms(), pdb.positions):
        if atom.residue.name != 'HOH':
            continue
        atom_list.append(atom.element.symbol)
        coor = coor.value_in_unit(unit.angstrom)
        coord_list.append([coor[0], coor[1], coor[2]])

    ani_input['solvent_atoms'] = ''.join(atom_list)
    ani_input['solvent_coords'] = np.array(coord_list) * unit.angstrom
    ani_input['box_length'] = box_length

def generate_rdkit_mol(smiles:str) -> Chem.Mol:
    """
    Generates a rdkit mol object with 3D coordinates from smiles
    Parameters
    ----------
    smiles: smiles string
    
    Returns
    ----------
    rdkit mol
    """
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m)
    return m


def reduced_pot(E:float) -> float:
    """
    Convert unit'd energy into a unitless reduced potential energy.

    In NVT:
        u(x) = U(x) / kBT
    """
    return E / kT



