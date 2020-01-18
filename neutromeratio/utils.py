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
            stereo_tag = 'STEREOE'
            nr_of_stereobonds += 1
        elif b.GetStereo() == Chem.rdchem.BondStereo.STEREOZ:
            stereo_tag = 'STEREOZ'
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

    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)

    stereo_tag = ''
    for b in m.GetBonds():
        if b.GetStereo() == Chem.rdchem.BondStereo.STEREOE:
            if stereo_tag:
                raise RuntimeError('There are multiple stereotags defined in this molecule')
            stereo_tag = 'STEREOE'
        elif b.GetStereo() == Chem.rdchem.BondStereo.STEREOZ:
            if stereo_tag:
                raise RuntimeError('There are multiple stereotags defined in this molecule')
            stereo_tag = 'STEREOZ'

    return stereo_tag


def generate_tautomer_class_stereobond_aware(name: str, t1_smiles: str, t2_smiles: str, nr_of_conformations: int=1) -> list:
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
    tautomers: list
        a list of tautomer(s)
    """
    

    tautomers = []
    from neutromeratio import Tautomer
    
    if get_nr_of_stereobonds(t1_smiles) == get_nr_of_stereobonds(t2_smiles):
        if get_nr_of_stereobonds(t1_smiles) == 0 and get_nr_of_stereobonds(t2_smiles) == 0:
            # no stereobond -- normal protocol
            # generate both rdkit mol
            logger.info('No stereobonds ...')
            tautomers.append(Tautomer(name=name, 
                                            initial_state_mol=generate_rdkit_mol(t1_smiles), 
                                            final_state_mol=generate_rdkit_mol(t2_smiles), 
                                            nr_of_conformations=nr_of_conformations))
        else:
            # stereobonds on both endstates
            # we need to add a torsion bias to make sure that the lambda protocol stopp at the correct torsion

            raise RuntimeError()
    else:
        if get_nr_of_stereobonds(t1_smiles) > get_nr_of_stereobonds(t2_smiles):
            
            t1_smiles_kappa_0 = t1_smiles
            t1_smiles_kappa_1 = change_only_stereotag(t1_smiles)
            tautomers.append(Tautomer(name=name, 
                                            initial_state_mol=generate_rdkit_mol(t1_smiles_kappa_0), 
                                            final_state_mol=generate_rdkit_mol(t2_smiles), 
                                            nr_of_conformations=nr_of_conformations))
            tautomers.append(Tautomer(name=name, 
                                            initial_state_mol=generate_rdkit_mol(t1_smiles_kappa_1), 
                                            final_state_mol=generate_rdkit_mol(t2_smiles), 
                                            nr_of_conformations=nr_of_conformations))
            
        elif get_nr_of_stereobonds(t1_smiles) < get_nr_of_stereobonds(t2_smiles):
            
            t1_smiles_kappa_0 = t2_smiles
            t1_smiles_kappa_1 = change_only_stereotag(t2_smiles)
            t2_smiles = t1_smiles
            tautomers.append(Tautomer(name=name, 
                                            initial_state_mol=generate_rdkit_mol(t1_smiles_kappa_0), 
                                            final_state_mol=generate_rdkit_mol(t2_smiles), 
                                            nr_of_conformations=nr_of_conformations))
            tautomers.append(Tautomer(name=name, 
                                            initial_state_mol=generate_rdkit_mol(t1_smiles_kappa_1), 
                                            final_state_mol=generate_rdkit_mol(t2_smiles), 
                                            nr_of_conformations=nr_of_conformations))
            

        else:
            raise RuntimeError()

    return tautomers
 



def find_torsion_idx(mol: Chem.Mol) -> list:

    a1_idx, a2_idx, a3_idx, a4_idx = (0, 0, 0, 0)
    a1, a2, a3, a4 = (None, None, None, None)

    for bond in mol.GetBonds():
        if str(bond.GetStereo()) == 'STEREONONE':
            continue
        else:
            a1_idx = bond.getBeginAtomIdx()
            a1 = bond.GetBeginAtom()

            a2_idx = bond.getEndAtomIdx()
            a2 = bond.GetEndAtom()
            break

    if not a1:
        raise RuntimeError(f"It seems as if there is not stereobond in the molecule.")

    a1_neighbors = a1.atom.GetNeighbors()
    a2_neighbors = a2.atom.GetNeighbors()


def change_only_stereotag(smiles: str) -> str:

    mol = Chem.MolFromSmiles(smiles)
    for bond in mol.GetBonds():
        if str(bond.GetStereo()) == 'STEREONONE':
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


def generate_torsion_restraint_from_minimized_conformation(tautomer, idx: list, lambda_value: float = 0.0):

    from .ani import LinearAlchemicalSingleTopologyANI, ANI1_force_and_energy
    from .restraints import TorsionHarmonicRestraint
    from .constants import device
    import torch

    alchemical_atoms = [tautomer.hybrid_hydrogen_idx_at_lambda_1,
                        tautomer.hybrid_hydrogen_idx_at_lambda_0]

    # extract hydrogen donor idx and hydrogen idx for from_mol
    model = LinearAlchemicalSingleTopologyANI(alchemical_atoms=alchemical_atoms)
    model = model.to(device)
    torch.set_num_threads(1)

    # perform initial sampling
    energy_function = ANI1_force_and_energy(
        model=model,
        atoms=tautomer.hybrid_atoms,
        mol=None,
        per_atom_thresh=0.4 * unit.kilojoule_per_mole,
        adventure_mode=True
    )

    for r in tautomer.ligand_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)

    for r in tautomer.hybrid_ligand_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)

    x0 = tautomer.hybrid_coords
    x0, _ = energy_function.minimize(x0,
                                     maxiter=100,
                                     lambda_value=lambda_value,
                                     kappa_value=0.0,
                                     show_plot=False)

    x0 = [x0.value_in_unit(unit.nanometer)]
    ani_traj = md.Trajectory(x0, tautomer.hybrid_topology)
    torsion = md.compute_dihedrals(ani_traj, [idx]) * unit.radian
    torsion = torsion.value_in_unit(unit.degree)
    logger.info(f"Torsion restraint will be around: {torsion}")

    return TorsionHarmonicRestraint(sigma=10 * unit.degree, atom_idx=[1, 2, 3, 4], torsion_angle=torsion[0] * unit.degree, active_at=0)
