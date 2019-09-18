import logging, copy, os, random
import mdtraj as md
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from .vis import display_mol
from .restraints import Restraint
from simtk import unit
from collections import namedtuple
from .conformations import generate_conformations_from_mol
from .hybrid import generate_hybrid_structure
from .utils import write_pdb
from .utils import add_solvent
import numpy as np

logger = logging.getLogger(__name__)

def get_tautomer_transformation(m1:Chem.Mol, m2:Chem.Mol)->namedtuple:
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
    atoms = '' # atom element string
    for a in m1.GetAtoms():
        atoms += str(a.GetSymbol())

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
    
    r1 = Restraint( sigma=0.1 * unit.angstrom, atom_i_idx=donor, atom_j_idx=hydrogen_idx_that_moves, atoms=atoms, active_at_lambda=1)
    r2 = Restraint( sigma=0.1 * unit.angstrom, atom_i_idx=acceptor, atom_j_idx=hydrogen_idx_that_moves, atoms=atoms, active_at_lambda=0)
   
    return { 'donor_idx': donor, 'hydrogen_idx' : hydrogen_idx_that_moves, 'acceptor_idx' : acceptor, 'restraints' : [r1,r2]}


def from_mol_to_ani_input(mol:Chem.Mol, nr_of_conf:int, in_solvent:bool=False)->dict:
    """
    Generates atom_list and coord_list entries from rdkit mol.
    Parameters
    ----------
    mol : rdkit.Chem.Mol
    nr_of_conf : int

    Returns
    -------
    { 'ligand_atoms' : atoms, 'ligand_coords' : coord_list} 
    """
    
    # generate atom list
    atom_list = []
    for a in mol.GetAtoms():
        atom_list.append(a.GetSymbol())

    # generate coord list
    coord_list = []
    mol = generate_conformations_from_mol(mol, nr_of_conf)

    for conf_idx in range(mol.GetNumConformers()):
        tmp_coord_list = []
        for a in mol.GetAtoms():
            pos = mol.GetConformer(conf_idx).GetAtomPosition(a.GetIdx())
            tmp_coord_list.append([pos.x, pos.y, pos.z])
        tmp_coord_list = np.array(tmp_coord_list) * unit.angstrom
        coord_list.append(tmp_coord_list)

    n = random.random()
    # TODO: use tmpfile for this https://stackabuse.com/the-python-tempfile-module/
    _ = write_pdb(mol, f"tmp{n:0.9f}.pdb")
    topology = md.load(f"tmp{n:0.9f}.pdb").topology
    os.remove(f"tmp{n:0.9f}.pdb")
    
    ani_input =  {'ligand_atoms' : ''.join(atom_list), 
            'ligand_coords' : coord_list, 
            'ligand_topology' : topology }

   
    if in_solvent:
        add_solvent(None, None, None, None)
    
    return ani_input