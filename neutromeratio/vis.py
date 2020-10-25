import logging
import os

import mdtraj as md
import nglview
import numpy as np
from IPython.core.display import display
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from simtk import unit

logger = logging.getLogger(__name__)


def generate_nglview_object(
    traj: md.Trajectory,
    radius: int = 0.0,
    donor_hydrogen_idx: int = -1,
    acceptor_hydrogen_idx: int = -1,
    mark_residues=[],
) -> nglview.NGLWidget:
    """
    Generates nglview object from a trajectory object. Generated with md.Trajectory(traj, top).
    Parameters
    ----------
    traj : md.Trajectory
    donor_hydrogen_idx : int
    acceptor_hydrogen_idx : int

    Returns
    -------
    view: nglview object
    """

    view = nglview.show_mdtraj(traj)
    view.add_representation(
        repr_type="ball+stick", selection="water", opacity=0.4, color="blue"
    )
    if radius > 0.0:
        view.shape.add_sphere(
            [radius / 2, radius / 2, radius / 2], [0, 0, 1], (radius) / 2
        )
        view.update_representation(component=1, repr_index=0, opacity=0.2)

    if donor_hydrogen_idx != -1 and acceptor_hydrogen_idx != -1:

        # Clear all representations to try new ones
        print("Hydrogen in GREEN  is real at lambda: 0.")
        print("Hydrogen in YELLOW is real at lambda: 1.")
        view.add_representation(
            "point", selection=[donor_hydrogen_idx], color="green", pointSize=3.5
        )
        view.add_representation(
            "point", selection=[acceptor_hydrogen_idx], color="yellow", pointSize=3.5
        )
    return view


def display_mol(mol: Chem.Mol):
    """
    Gets mol as input and displays its 2D Structure using IPythonConsole.
    """

    def mol_with_atom_index(mol):
        atoms = mol.GetNumAtoms()
        for idx in range(atoms):
            mol.GetAtomWithIdx(idx).SetProp(
                "molAtomMapNumber", str(mol.GetAtomWithIdx(idx).GetIdx())
            )
        return mol

    mol = mol_with_atom_index(mol)
    AllChem.Compute2DCoords(mol)
    display(mol)
