# Add imports here
from simtk import unit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
import copy
import os
import mdtraj as md
import numpy as np
from .constants import kT
import nglview
import logging
from rdkit.Chem.Draw import IPythonConsole
from IPython.core.display import display
from pdbfixer import PDBFixer
from simtk.openmm import Vec3
import random
from .ani import ANI1_force_and_energy
import shutil

logger = logging.getLogger(__name__)

    
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

def display_mol(mol:Chem.Mol):
    """
    Gets mol as input and displays its 2D Structure using IPythonConsole.
    """

    def mol_with_atom_index(mol):
        atoms = mol.GetNumAtoms()
        for idx in range( atoms ):
            mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
        return mol

    mol = mol_with_atom_index(mol)
    AllChem.Compute2DCoords(mol)
    display(mol)


def generate_nglview_object(traj:md.Trajectory, tautomer_transformation:dict) -> nglview.NGLWidget:
    """
    Generates nglview object from topology and trajectory files.
    Parameters
    ----------
    top_file : file path to mdtraj readable topology file
    traj_file : file path to mdtraj readable trajectory file

    Returns
    -------
    view: nglview object
    """

    view = nglview.show_mdtraj(traj)
    if 'donor_hydrogen_idx' in tautomer_transformation:

        # Clear all representations to try new ones
        print('Hydrogen in GREEN  is real at lambda: 0.')
        print('Hydrogen in YELLOW is real at lambda: 1.')
        view.add_representation('point', selection=[tautomer_transformation['donor_hydrogen_idx']], color='green', pointSize=3.5)
        view.add_representation('point', selection=[tautomer_transformation['acceptor_hydrogen_idx']], color='yellow', pointSize=3.5)
    
    return view 


def from_mol_to_ani_input(mol: Chem.Mol) -> dict:
    """
    Generates atom_list and coord_list entries from rdkit mol.
    Parameters
    ----------
    mol : rdkit.Chem.Mol

    Returns
    -------
    { 'ligand_atoms' : atoms, 'ligand_coords' : coord_list} 
    """
    
    atom_list = []
    coord_list = []
    for a in mol.GetAtoms():
        atom_list.append(a.GetSymbol())
        pos = mol.GetConformer().GetAtomPosition(a.GetIdx())
        coord_list.append([pos.x, pos.y, pos.z])

    n = random.random()
    # TODO: use tmpfile for this https://stackabuse.com/the-python-tempfile-module/
    _ = write_pdb(mol, f"tmp{n:0.9f}.pdb")
    topology = md.load(f"tmp{n:0.9f}.pdb").topology
    os.remove(f"tmp{n:0.9f}.pdb")
    
    logging.info('Initially generating {} conformations ...'.format(nr_of_conformations))
    mol, rmsd = _generate_conformations_from_mol(input_smi=smiles, nr_of_conformations=nr_of_conformations, molecule_name=name)


    return { 'ligand_atoms' : ''.join(atom_list), 
            'ligand_coords' : np.array(coord_list) * unit.angstrom, 
            'ligand_topology' : topology,
            }
