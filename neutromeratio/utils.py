import logging
import os
import random
from typing import Any, List, Tuple, Union

import mdtraj as md
import nglview
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from simtk import unit

StereoBondType = Union[Any, str]

from neutromeratio.constants import kT

logger = logging.getLogger(__name__)


def find_idx(query_name: str) -> list:
    from neutromeratio.constants import _get_names

    protocol = dict()
    idx = 1
    for mol_idx, name in enumerate(_get_names()):
        list_of_idx = []
        list_of_lambdas = []
        for lamb in np.linspace(0, 1, 11):
            list_of_lambdas.append(lamb)
            list_of_idx.append(idx)
            idx += 1
        protocol[name] = (list_of_idx, list_of_lambdas, mol_idx + 1)
    return protocol[query_name]


def write_pdb(mol: Chem.Mol, filepath: str, confId: int = -1) -> str:
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


def flag_unspec_stereo(smiles: str) -> bool:
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)

    unspec = False
    Chem.FindPotentialStereoBonds(m)
    for bond in m.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREOANY:
            logger.debug(
                bond.GetBeginAtom().GetSymbol(),
                bond.GetSmarts(),
                bond.GetEndAtom().GetSymbol(),
            )
            unspec = True
    return unspec


def decide_unspec_stereo(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)

    Chem.FindPotentialStereoBonds(m)
    for bond in m.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREOANY:
            print(
                bond.GetBeginAtom().GetSymbol(),
                bond.GetSmarts(),
                bond.GetEndAtom().GetSymbol(),
            )
            bond.SetStereo(Chem.BondStereo.STEREOE)
    return Chem.MolToSmiles(m)


def _get_traj(traj_path, top_path, remove_idx=[]):
    top = md.load(top_path).topology
    traj = md.load(traj_path, top=top)
    atoms = [a for a in range(top.n_atoms)]
    if remove_idx:
        for idx in remove_idx:
            atoms.remove(idx)
        print("Atoms that are not removed: {atoms}")
        traj = traj.atom_slice(atoms)
    return traj, top


def get_nr_of_stereobonds(smiles: str) -> int:
    """
    Calculates the nr of stereobonds.
    Parameters
    ----------
    smiles: a SMILES string.

    Returns
    ----------
    nr of stereobonds: int
    """

    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)

    nr_of_stereobonds = 0

    for b in m.GetBonds():
        if b.GetStereo() == Chem.rdchem.BondStereo.STEREOE:
            stereo_tag = "STEREOE"
            nr_of_stereobonds += 1
        elif b.GetStereo() == Chem.rdchem.BondStereo.STEREOZ:
            stereo_tag = "STEREOZ"
            nr_of_stereobonds += 1

    return nr_of_stereobonds


def get_stereotag_of_stereobonds(smiles: str) -> int:
    """
    Calculates the nr of stereobonds.
    Parameters
    ----------
    smiles: a SMILES string.

    Returns
    ----------
    stereotag: str
    """

    try:
        m = Chem.MolFromSmiles(smiles)
        m = Chem.AddHs(m)
    except TypeError:
        m = smiles
        pass

    stereo_tag = ""
    for b in m.GetBonds():
        if b.GetStereo() == Chem.rdchem.BondStereo.STEREOE:
            if stereo_tag:
                raise RuntimeError(
                    "There are multiple stereotags defined in this molecule"
                )
            stereo_tag = "STEREOE"
        elif b.GetStereo() == Chem.rdchem.BondStereo.STEREOZ:
            if stereo_tag:
                raise RuntimeError(
                    "There are multiple stereotags defined in this molecule"
                )
            stereo_tag = "STEREOZ"

    return stereo_tag


def generate_new_tautomer_pair(name: str, t1_smiles: str, t2_smiles: str):
    """Constructs and returns a Tautomer pair object, generating RDKit mols
    from t1 and t2 smiles"""
    # TOOD: should this also accept nr_of_conformations, enforceChirality?
    from neutromeratio.tautomers import Tautomer

    return Tautomer(
        name=name,
        initial_state_mol=generate_rdkit_mol(t1_smiles),
        final_state_mol=generate_rdkit_mol(t2_smiles),
        nr_of_conformations=1,
        enforceChirality=True,
    )


def generate_tautomer_class_stereobond_aware(
    name: str,
    t1_smiles: str,
    t2_smiles: str,
    nr_of_conformations: int = 1,
    enforceChirality=True,
) -> Tuple[StereoBondType, List, bool]:
    """
    If a stereobond is present in the tautomer pair we need to transform from the molecule
    with the stereobond (e.g. enol) to the tautomer without the stereobond (e.g. keto). This
    function makes sure that this happens and returns a list of tautomers.
    If there is no stereobond present in either tautomer, the list contains only one tautomer.
    If there is a stereobond present the list contains two tautomers (one with cis, one with trans configuration).

    Parameters
    ----------
    name: str
        The name of the tautomer
    t1_smiles: str
        a SMILES string
    t2_smiles: str
        a SMILES string
    nr_of_conformations: int
        nr of conformations

    Returns
    ----------
    stereobond_type: StereoBondType
        one of [None, "generic", "Imine"]
    tautomers: list
        a list of tautomer(s)
    flipped: bool
        to indicate that t1/t2 SMILES have been exchanged


    Notes
    -----
    one branch (for detecting stereobonds in heterocycles) depends on name
    """
    from neutromeratio.tautomers import Tautomer

    tautomers = []

    flipped = False
    stereobond_type = None

    def _tautomer(t1, t2):
        """Tautomer constructor, with name, nr_of_conformations, enforceChirality applied"""
        return Tautomer(name, t1, t2, nr_of_conformations, enforceChirality)

    if flag_unspec_stereo(t1_smiles) or flag_unspec_stereo(t2_smiles):
        logger.debug("Imines present ... switching to imine generation.")
        stereobond_type = "Imine"

        if flag_unspec_stereo(t1_smiles):
            t1_kappa_0 = change_stereobond_in_imine_to_cis(
                generate_rdkit_mol(t1_smiles)
            )
            t1_kappa_1 = change_stereobond_in_imine_to_trans(
                generate_rdkit_mol(t1_smiles)
            )

            t1_a = t1_kappa_0
            t2_a = generate_rdkit_mol(t2_smiles)

            t1_b = t1_kappa_1
            t2_b = generate_rdkit_mol(t2_smiles)

            tautomers.append(_tautomer(t1_a, t2_a))
            tautomers.append(_tautomer(t1_b, t2_b))

        elif flag_unspec_stereo(t2_smiles):
            flipped = True
            t2_kappa_0 = change_stereobond_in_imine_to_cis(
                generate_rdkit_mol(t2_smiles)
            )
            t2_kappa_1 = change_stereobond_in_imine_to_trans(
                generate_rdkit_mol(t2_smiles)
            )

            t1_a = t2_kappa_0
            t2_a = generate_rdkit_mol(t1_smiles)

            t1_b = t2_kappa_1
            t2_b = generate_rdkit_mol(t1_smiles)

            tautomers.append(_tautomer(t1_a, t2_a))
            tautomers.append(_tautomer(t1_b, t2_b))

        else:
            raise RuntimeError("Stereobonds present in both tautomers ... aborting!")

    elif get_nr_of_stereobonds(t1_smiles) == get_nr_of_stereobonds(t2_smiles):
        t1 = generate_rdkit_mol(t1_smiles)
        t2 = generate_rdkit_mol(t2_smiles)

        if (
            get_nr_of_stereobonds(t1_smiles) == 0
            and get_nr_of_stereobonds(t2_smiles) == 0
        ):
            # no stereobond -- normal protocol
            # generate both rdkit mol
            logger.debug("No stereobonds ...")
            tautomers.append(_tautomer(t1, t2))

        elif name == "molDWRow_1636":
            logger.debug("molDWRow_1636 -- stereobonds in hetereocycle ...")
            tautomers.append(_tautomer(t1, t2))

        else:
            # stereobonds on both endstates
            # we need to add a torsion bias to make sure that the lambda protocol stops at the correct torsion
            raise RuntimeError("Two stereobonds ... aborting")
    elif get_nr_of_stereobonds(t1_smiles) > get_nr_of_stereobonds(t2_smiles):
        stereobond_type = "generic"
        t1_smiles_kappa_0 = t1_smiles
        t1_smiles_kappa_1 = change_only_stereobond(t1_smiles)

        t1_a = generate_rdkit_mol(t1_smiles_kappa_0)
        t2_a = generate_rdkit_mol(t2_smiles)

        t1_b = generate_rdkit_mol(t1_smiles_kappa_1)
        t2_b = generate_rdkit_mol(t2_smiles)

        tautomers.append(_tautomer(t1_a, t2_a))
        tautomers.append(_tautomer(t1_b, t2_b))

    elif get_nr_of_stereobonds(t1_smiles) < get_nr_of_stereobonds(t2_smiles):
        stereobond_type = "generic"
        flipped = True
        t1_smiles_kappa_0 = t2_smiles
        t1_smiles_kappa_1 = change_only_stereobond(t2_smiles)
        t2_smiles = t1_smiles

        t1_a = generate_rdkit_mol(t1_smiles_kappa_0)
        t2_a = generate_rdkit_mol(t2_smiles)

        t1_b = generate_rdkit_mol(t1_smiles_kappa_1)
        t2_b = generate_rdkit_mol(t2_smiles)

        tautomers.append(_tautomer(t1_a, t2_a))
        tautomers.append(_tautomer(t1_b, t2_b))

    else:
        raise RuntimeError("Stereobonds present in both tautomers ... aborting!")

    return stereobond_type, tautomers, flipped


def change_stereobond_in_imine_to_trans(mol: Chem.Mol) -> Chem.Mol:

    Chem.FindPotentialStereoBonds(mol)
    for bond in mol.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREOANY:
            logger.debug(
                f"{bond.GetBeginAtom().GetSymbol()} {bond.GetSmarts()} {bond.GetEndAtom().GetSymbol()}"
            )
            bond.SetStereo(Chem.BondStereo.STEREOE)

    return mol


def change_stereobond_in_imine_to_cis(mol: Chem.Mol) -> Chem.Mol:

    Chem.FindPotentialStereoBonds(mol)
    for bond in mol.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREOANY:
            logger.debug(
                f"{bond.GetBeginAtom().GetSymbol()} {bond.GetSmarts()} {bond.GetEndAtom().GetSymbol()}"
            )
            bond.SetStereo(Chem.BondStereo.STEREOZ)

    return mol


def change_only_stereobond(smiles: str) -> str:

    mol = Chem.MolFromSmiles(smiles)
    for bond in mol.GetBonds():
        if str(bond.GetStereo()) == "STEREONONE":
            continue
        else:
            # found one, reverse the stereocenter and generate conformations
            logger.debug(f"Stereobond that will be reversed: {bond.GetStereo()}")
            if bond.GetStereo() == Chem.rdchem.BondStereo.STEREOE:
                bond.SetStereo(Chem.rdchem.BondStereo.STEREOZ)
            elif bond.GetStereo() == Chem.rdchem.BondStereo.STEREOZ:
                bond.SetStereo(Chem.rdchem.BondStereo.STEREOE)

    return Chem.MolToSmiles(mol)


def generate_rdkit_mol(smiles: str) -> Chem.Mol:
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
    AllChem.EmbedMolecule(m, enforceChirality=True)
    return m


def reduced_pot(E: float) -> float:
    """
    Convert unit'd energy into a unitless reduced potential energy.

    In NVT:
        u(x) = U(x) / kBT
    """
    return E / kT
