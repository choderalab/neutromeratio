# Add imports here
from simtk import unit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
import copy
import os
import mdtraj as md
import numpy as np
from .config import kT
import nglview

def get_donor_atom_idx(m1:Chem.Mol, m2:Chem.Mol) -> dict:
    """
    Returns the atom index of the hydrogen donor atom and hydrogen atom that moves.
    This index is consistent with the indexing of m1.
    Parameters
    ----------
    m1: rdkit mol object
    m2: rdkit mol object
    
    Returns
    -------
    dict('donor': donor_idx, 'hydrogen_idx' : hydrogen_idx_that_moves)
    """

    m1 = copy.deepcopy(m1)
    m2 = copy.deepcopy(m2)
    # find substructure and generate mol from substructure
    sub = rdFMCS.FindMCS([m1, m2], bondCompare=Chem.rdFMCS.BondCompare.CompareOrder.CompareAny)
    mcsp = Chem.MolFromSmarts(sub.smartsString, False)
    g = Chem.MolFromSmiles(Chem.MolToSmiles(mcsp, allHsExplicit=True), sanitize=False)
    substructure_idx_m1 = m1.GetSubstructMatch(g)

    #get idx of hydrogen that moves to new position
    hydrogen_idx_that_moves = -1
    for a in m1.GetAtoms():
        if a.GetIdx() not in substructure_idx_m1:
            print('m1: Index of atom that moves: {}.'.format(a.GetIdx()))
            hydrogen_idx_that_moves = a.GetIdx()

    # get idx of connected heavy atom which is the donor atom
    # there can only be one neighbor, therefor it is valid to take the first neighbor of the hydrogen
    donor = int(m1.GetAtomWithIdx(hydrogen_idx_that_moves).GetNeighbors()[0].GetIdx())
    return { 'donor': donor, 'hydrogen_idx' : hydrogen_idx_that_moves }


def write_xyz_traj_file(atoms:str, coordinates:np.array, name:str='test'):
    """
    Writes xyz trajectory file in current directory.
    The trajectory is saved in traj_{name}.xyz.
    Parameters
    ----------
    atoms: list of atoms (in a single string) 
    coordinates: numpy array with coordinates
    name: name of the traj
    """
    if os.path.exists('traj_{}.xyz'.format(name)):
        f = open('traj_{}.xyz'.format(name), 'a')
    else:
        f = open('traj_{}.xyz'.format(name), 'w')

    for frame in coordinates:
        frame_in_angstrom = frame.value_in_unit(unit.angstrom)
        f.write('{}\n'.format(len(atoms)))
        f.write('{}\n'.format('writing mols'))
        for atom, coordinate in zip(atoms, frame_in_angstrom):
            f.write('  {:2}   {: 11.9f}  {: 11.9f}  {: 11.9f}\n'.format(atom, coordinate[0], coordinate[1], coordinate[2]))

def write_xyz_file(atom_list:str, coordinates:np.array, name:str='test', identifier:str='0_0'):
    """
    Writes xyz file in ./mc_confs directory. If directory does not exist it is created.
    The file is saved in {name}_{identifier}.xyz.
    Parameters
    ----------
    atoms: list of atoms (in a single string) 
    coordinates: numpy array with coordinates
    name: name of the file
    """
    
    if not os.path.exists('mc_confs'):
        os.mkdir('mc_confs')

    f = open('mc_confs/{}_{}.xyz'.format(name, identifier), 'w')
    f.write('{}\n'.format(len(atom_list)))
    f.write('{}\n'.format('writing mols'))
    coordinates_in_angstroms = coordinates.value_in_unit(unit.angstrom)
    for atom, coordinate in zip(atom_list, coordinates_in_angstroms):
        f.write('  {:2}   {: 11.9f}  {: 11.9f}  {: 11.9f}\n'.format(atom, coordinate[0], coordinate[1], coordinate[2]))

def write_pdb(mol:Chem.Mol, path:str, name:str, tautomer_id:int) -> str:
    """
    Writes pdb file in path directory. If directory does not exist it is created.
    The file is saved in {path}/{name}_{tautomer_id}.pdb.
    Parameters
    ----------
    mol: the mol that should be saved.
    path: the location the pdb file should be saved
    name: name of the file
    tautomer_id: tautomer id

    Returns
    ----------
    PDBfile as string
    """

    if not os.path.exists(path):
        os.mkdir(path)

    Chem.MolToPDBFile(mol, '{}/{}_{}.pdb'.format(path, name, tautomer_id))
    return Chem.MolToPDBBlock(mol)

def generate_rdkit_mol(smiles:str) -> Chem.Mol:
    """
    Geneartes a rdkit mol object with 3D coordinates from smiles
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


def from_mol_to_ani_input(mol: Chem.Mol) -> dict:
    # generates atom_list and coord_list entries from rdkit mol
    atom_list = []
    coord_list = []
    for a in mol.GetAtoms():
        atom_list.append(a.GetSymbol())
        pos = mol.GetConformer().GetAtomPosition(a.GetIdx())
        coord_list.append([pos.x, pos.y, pos.z])
    return { 'atom_list' : ''.join(atom_list), 'coord_list' : coord_list}


def reduced_pot(E:float) -> float:
    """
    Convert unit'd energy into a unitless reduced potential energy.

    In NVT:
        u(x) = U(x) / kBT
    """
    return E / kT

def display_mol(mol):
    from rdkit.Chem.Draw import IPythonConsole
    def mol_with_atom_index(mol):
        atoms = mol.GetNumAtoms()
        for idx in range( atoms ):
            mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
        return mol

    mol = mol_with_atom_index(mol)
    AllChem.Compute2DCoords(mol)
    display(mol)


def generate_nglview_object(top_file:str, traj_file:str) -> nglview.NGLWidget:
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

    topology = md.load(top_file).topology
    ani_traj = md.load(traj_file, top=topology)

    return nglview.show_mdtraj(ani_traj)
