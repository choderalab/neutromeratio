import logging
import os
import random

import mdtraj as md
import nglview
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from simtk import unit

from .constants import kT

logger = logging.getLogger(__name__)

def write_pdb(mol:Chem.Mol, filepath:str, confId:int=-1)->str:
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

    Chem.MolToPDBFile(mol, filepath, confId)
    return Chem.MolToPDBBlock(mol)

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
