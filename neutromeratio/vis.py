# Add imports here
from simtk import unit
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import mdtraj as md
import numpy as np
import nglview
import logging
from rdkit.Chem.Draw import IPythonConsole
from IPython.core.display import display

logger = logging.getLogger(__name__)

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
