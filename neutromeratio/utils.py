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

    

def get_tautomer_transformation(m1:Chem.Mol, m2:Chem.Mol) -> dict:
    """
    Returns the atom index of the hydrogen donor atom and hydrogen atom that moves.
    This index is consistent with the indexing of m1.
    Parameters
    ----------
    m1: rdkit mol object
    m2: rdkit mol object
    
    Returns
    -------
    { 'donor_idx': donor, 'hydrogen_idx' : hydrogen_idx_that_moves, 'acceptor_idx' : acceptor}
    """

    m1 = copy.deepcopy(m1)
    m2 = copy.deepcopy(m2)
    # find substructure and generate mol from substructure
    sub_m = rdFMCS.FindMCS([m1, m2], bondCompare=Chem.rdFMCS.BondCompare.CompareOrder.CompareAny)
    mcsp = Chem.MolFromSmarts(sub_m.smartsString, False)

    # the order of the substructure lists are the same for both 
    # substructure matches => substructure_idx_m1[i] = substructure_idx_m2[i]
    substructure_idx_m1 = m1.GetSubstructMatch(mcsp)
    substructure_idx_m2 = m2.GetSubstructMatch(mcsp)

    #get idx of hydrogen that moves to new position
    hydrogen_idx_that_moves = -1
    for a in m1.GetAtoms():
        if a.GetIdx() not in substructure_idx_m1:
            logger.info('Index of atom that moves: {}.'.format(a.GetIdx()))
            hydrogen_idx_that_moves = a.GetIdx()

    # get idx of connected heavy atom which is the donor atom
    # there can only be one neighbor, therefor it is valid to take the first neighbor of the hydrogen
    donor = int(m1.GetAtomWithIdx(hydrogen_idx_that_moves).GetNeighbors()[0].GetIdx())
    logger.info('Index of atom that donates hydrogen: {}'.format(donor))

    logging.debug(substructure_idx_m1)
    logging.debug(substructure_idx_m2)
    for i in range(len(substructure_idx_m1)):
        a1 = m1.GetAtomWithIdx(substructure_idx_m1[i])
        if a1.GetSymbol() != 'H':
            a2 = m2.GetAtomWithIdx(substructure_idx_m2[i])
            # get acceptor - there are two heavy atoms that have 
            # not the same number of neighbors
            a1_neighbors = a1.GetNeighbors()
            a2_neighbors = a2.GetNeighbors()
            acceptor_count = 0
            if (len(a1_neighbors)) != (len(a2_neighbors)):
                # we are only interested in the one that is not already the donor
                if substructure_idx_m1[i] == donor:
                    continue
                acceptor = substructure_idx_m1[i]
                logger.info('Index of atom that accepts hydrogen: {}'.format(acceptor))
                acceptor_count += 1
                if acceptor_count > 1:
                    raise RuntimeError('There are too many potential acceptor atoms.')

    AllChem.Compute2DCoords(m1)
    display_mol(m1)
    return { 'donor_idx': donor, 'hydrogen_idx' : hydrogen_idx_that_moves, 'acceptor_idx' : acceptor}


def generate_xyz_string(atom_str:str, coordinates:unit.quantity.Quantity)->str:
    """
    Returns xyz file as string.
    Parameters
    ----------
    atoms: list of atoms (in a single string) 
    coordinates: numpy array with coordinates
    """
    s = '{}\n'.format(len(atom_str))
    s += '{}\n'.format('writing mols')

    coordinates_in_angstroms = coordinates.value_in_unit(unit.angstrom)
    for atom, coordinate in zip(atom_str, coordinates_in_angstroms):
        s += '  {:2}   {: 11.9f}  {: 11.9f}  {: 11.9f}\n'.format(atom, coordinate[0], coordinate[1], coordinate[2])
    
    return s

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


def generate_nglview_object(top_file:str, traj_files:list) -> nglview.NGLWidget:
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
    ani_traj = md.load(traj_files, top=topology)

    return nglview.show_mdtraj(ani_traj)

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

    _ = write_pdb(mol, 'tmp.pdb')
    topology = md.load('tmp.pdb').topology
    os.remove('tmp.pdb')
    
    return { 'ligand_atoms' : ''.join(atom_list), 
            'ligand_coords' : np.array(coord_list) * unit.angstrom, 
            'ligand_topology' : topology,
            }


class MonteCarloBarostat(object):

    def __init__(self, pbc_box_length:unit.Quantity, energy:ANI1_force_and_energy):

        assert(type(pbc_box_length) == unit.Quantity)       
        self.current_volumn = pbc_box_length ** 3
        self.num_attempted = 0
        self.num_accepted = 0
        self.volume_scale = 0.01 * self.current_volumn
        self.energy_function = energy

    def update_volumn(self, x:unit.Quantity):

        assert(type(x) == unit.Quantity)       
        print(self.energy_function.model.pbc)
        energy = self.energy_function.calculate_energy(x)
        current_volumn = self.current_volumn
        delta_volumn = current_volumn * 2 * (random.uniform(0, 1) - 0.5)
        new_volumn = current_volumn + delta_volumn
        length_scale = (new_volumn/current_volumn) ** (1.0/3.0)




