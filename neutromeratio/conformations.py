import logging, copy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS

logger = logging.getLogger(__name__)


def generate_conformations_from_mol(mol: Chem.Mol, nr_of_conformations:int=100, molecule_name:str = None):
    """
    Generates a rdkit molecule from a SMILES string, generates conformations and generates a dictionary representation of it.
    
    Keyword arguments:
    input_smi: SMILES string
    nr_of_conformations: int
    molecule_name [optional]: String
    """  

    charge = 0
    for at in mol.GetAtoms():
        if at.GetFormalCharge() != 0:
            charge += int(at.GetFormalCharge())          

    if charge != 0:
        logger.warning('Charged system')
        logger.info(Chem.MolToSmiles(mol))

    Chem.rdmolops.RemoveStereochemistry(mol)
    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)

    mol.SetProp("smiles", Chem.MolToSmiles(mol))
    mol.SetProp("charge", str(charge))
    
    if molecule_name:
        mol.SetProp("name", str(molecule_name))

    # generate numConfs for the smiles string 
    Chem.rdDistGeom.EmbedMultipleConfs(mol, numConfs=nr_of_conformations, enforceChirality=False, pruneRmsThresh=0.1)
    # aligne them and minimize
    AllChem.AlignMolConformers(mol)
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    return mol

