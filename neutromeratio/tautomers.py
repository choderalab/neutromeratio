import logging, copy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import IPythonConsole
from IPython.core.display import display
from .utils import display_mol

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
