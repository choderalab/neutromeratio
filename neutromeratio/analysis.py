import copy
import logging
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from scipy.special import logsumexp
from simtk import unit
import mdtraj as md
import torchani
import torch
from rdkit.Chem import rdFMCS
import pkg_resources

from .constants import kT, gas_constant, temperature, mols_with_charge, exclude_set_ANI, multiple_stereobonds, device
from .tautomers import Tautomer
from .parameter_gradients import FreeEnergyCalculator
from .utils import generate_tautomer_class_stereobond_aware
from .ani import LinearAlchemicalSingleTopologyANI, ANI1_force_and_energy
from glob import glob


logger = logging.getLogger(__name__)

def _remove_hydrogens(m: Chem.Mol):
    """Removing all hydrogens from the molecule with a few exceptions"""
    #########################################
    # search for important hydrogens
    #########################################
    keep_hydrogens = []
    logger.debug('search for patterns ...')
    # test for primary alcohol
    patt = Chem.MolFromSmarts('[OX2H]')
    if m.HasSubstructMatch(patt):
        logger.debug('found primary alcohol')
        l = m.GetSubstructMatch(patt)
        keep_hydrogens.extend(l)

    # test for imine
    patt = Chem.MolFromSmarts('[CX3]=[NH]')
    if m.HasSubstructMatch(patt):
        logger.debug('found imine')
        l = m.GetSubstructMatch(patt)
        keep_hydrogens.extend(l)

    # test for primary amine
    patt = Chem.MolFromSmarts('[NX3;H2]')
    if m.HasSubstructMatch(patt):
        logger.debug('found primary amine')
        l = m.GetSubstructMatch(patt)
        keep_hydrogens.extend(l)

    # test for secondary amine
    patt = Chem.MolFromSmarts('[NX3H]')
    if m.HasSubstructMatch(patt):
        logger.debug('found secondary amine')
        l = m.GetSubstructMatch(patt)
        keep_hydrogens.extend(l)

    # test for cyanamide
    patt = Chem.MolFromSmarts('[NX3][CX2]#[NX1]')
    if m.HasSubstructMatch(patt):
        logger.debug('found cyanamide')
        l = m.GetSubstructMatch(patt)
        keep_hydrogens.extend(l)

    # test for thiol
    patt = Chem.MolFromSmarts('[#16X2H]')
    if m.HasSubstructMatch(patt):
        logger.debug('found thiol')
        l = m.GetSubstructMatch(patt)
        keep_hydrogens.extend(l)

    # unfortunatelly, RemoveHs() does not retain marked hydrogens
    # therefore, mutating important Hs to Li and then remutating them
    # to Hs
    if keep_hydrogens:
        for idx in keep_hydrogens:
            atom = m.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'H':
                    hydrogen = m.GetAtomWithIdx(neighbor.GetIdx())
                    hydrogen.SetAtomicNum(3)

        m = Chem.RemoveHs(m)
        for atom in m.GetAtoms():
            if atom.GetSymbol() == 'Li':
                hydrogen = m.GetAtomWithIdx(atom.GetIdx())
                hydrogen.SetAtomicNum(1)
    else:
        m = Chem.RemoveHs(m)

    return m


def prune_conformers(mol: Chem.Mol, energies: list, rmsd_threshold: float) -> (Chem.Mol, list):
    """
    Adopted from: https://github.com/skearnes/rdkit-utils/blob/master/rdkit_utils/conformers.py
    Prune conformers from a molecule using an RMSD threshold, starting
    with the lowest energy conformer.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    Returns
    -------
    A new RDKit Mol containing the chosen conformers, sorted by
    increasing energy.
    """
    from typing import List
    rmsd = get_conformer_rmsd(mol)
    sort = np.argsort([x.value_in_unit(unit.kilocalorie_per_mole)
                       for x in energies])  # sort by increasing energy
    keep:List[int] = []  # always keep lowest-energy conformer
    discard:List[int] = []
    for i in sort:

        # always keep lowest-energy conformer
        if len(keep) == 0:
            keep.append(i)
            continue

        # get RMSD to selected conformers
        this_rmsd = rmsd[i][np.asarray(keep, dtype=int)]

        # discard conformers within the RMSD threshold
        if np.all(this_rmsd >= rmsd_threshold):
            keep.append(i)
        else:
            discard.append(i)

    # create a new molecule to hold the chosen conformers
    # this ensures proper conformer IDs and energy-based ordering
    new_mol = Chem.Mol(mol)
    new_mol.RemoveAllConformers()
    conf_ids = [conf.GetId() for conf in mol.GetConformers()]
    filtered_energies = []
    print(f"keep: {keep}")
    for i in keep:
        print(i)
        conf = mol.GetConformer(conf_ids[i])
        filtered_energies.append(energies[i])
        new_mol.AddConformer(conf, assignId=True)
    return new_mol, filtered_energies


def get_conformer_rmsd(mol) -> list:
    """
    Calculate conformer-conformer RMSD.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    rmsd = np.zeros((mol.GetNumConformers(), mol.GetNumConformers()),
                    dtype=float)

    mol = _remove_hydrogens(copy.deepcopy(mol))
    for i, ref_conf in enumerate(mol.GetConformers()):
        for j, fit_conf in enumerate(mol.GetConformers()):
            if i >= j:
                continue
            rmsd[i, j] = AllChem.GetBestRMS(mol, mol, ref_conf.GetId(),
                                            fit_conf.GetId())
            rmsd[j, i] = rmsd[i, j]
    return rmsd


def calculate_weighted_energy(e_list):
    # G = -RT ln Σ exp(-G/RT)
    e_bw = (-1) * gas_constant * temperature * (logsumexp([(-1) * (e) / (gas_constant * temperature) for e in e_list] ))
    return e_bw


def _mol_to_nx(mol: Chem.Mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())

    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G


def ligand_degeneracy(mol):
    # Now iterate over graph isomorphisms
    from networkx.algorithms.isomorphism import GraphMatcher
    G = _mol_to_nx(mol)
    graph_matcher = GraphMatcher(G, G)
    degeneracy = sum([1 for isomorphism in graph_matcher.match()])
    return degeneracy


def entropy_correction(mol):
    return (- kT * np.log(ligand_degeneracy(mol))).in_units_of(unit.kilocalorie_per_mole)

def compare_confomer_generator_and_trajectory_minimum_structures(results_path: str, name: str, base: str, tautomer_idx: int, thinning:int = 100):
    assert (tautomer_idx == 1 or tautomer_idx == 2)

    ani_results = pickle.load(open(f'{results_path}/ani_mm_results.pickle', 'rb'))
    exp_results = pickle.load(open(f'{results_path}/exp_results.pickle', 'rb'))

    # generate the tautomer object
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']
    t_type, tautomers, flipped = generate_tautomer_class_stereobond_aware(name,
                                                                        t1_smiles,
                                                                        t2_smiles,
                                                                        nr_of_conformations=1,
                                                                        enforceChirality=True)


    tautomer = tautomers[0]
    print(f'Flipped: {flipped}')
    tautomer.perform_tautomer_transformation()

    tautomer_mol = (prune_conformers(ani_results[name]['t1-confs'], ani_results[name]['t1-energies'], rmsd_threshold=0.1))
    print(len(tautomer_mol[1]))

    traj_path = f'{base}/{name}/{name}_lambda_{tautomer_idx-1}.0000_kappa_0.0000_in_vacuum.dcd'
    pdb_path = f'{base}/{name}/{name}_0.pdb'

    # load trajectory, remove dummy atom
    traj = md.load(traj_path, top=pdb_path)
    atom_idx = [a.index for a in traj.topology.atoms]
    if (tautomer_idx -1) == 1:
        atom_idx.remove(int(tautomer.hybrid_hydrogen_idx_at_lambda_0))
    else:
        atom_idx.remove(int(tautomer.hybrid_hydrogen_idx_at_lambda_1))

    traj = traj.atom_slice(atom_indices=atom_idx)

    # save pdb without dummy atom
    tautomer_pdb = f'{base}/{name}/{name}_without_dummy_{tautomer_idx}.pdb'
    traj[0].save_pdb(tautomer_pdb)

    # generate rdkit mol object with the same atom indizes as the trajectory but without the dummy atom
    mol = Chem.MolFromPDBFile(tautomer_pdb, removeHs=False)
    # remove conf of pdb
    mol.RemoveAllConformers()

    # generate energy function, use atom symbols of rdkti mol
    from .ani import PureANI1ccx, ANI1_force_and_energy
    model = PureANI1ccx()
    energy_function = ANI1_force_and_energy(model=model,
                                            atoms=[a.GetSymbol() for a in mol.GetAtoms()],
                                            mol=None)

    # take every 100th conformation and minimize it using ANI1
    minimized_traj = [] # store min conformations in here

    for idx, conf in enumerate(traj[::thinning]):
       
        print(f'{idx}/{len(traj[::thinning])}')
        c = (conf.xyz[0]) * unit.nanometer
        min_conf = energy_function.minimize(c)[0] # only real atoms, therefor lambda not needed
        minimized_traj.append(min_conf)
        new_conf = _generate_conformer(min_conf)
        # add the conformation to the rdkit mol object
        mol.AddConformer(new_conf, assignId=True)

    # generate mdtraj object with minimized confs
    minimum_traj = md.Trajectory(np.array([v.value_in_unit(unit.nanometer) for v in minimized_traj]), traj.topology)

    # generate reference_mol
    reference = prune_conformers(ani_results[name][f't{tautomer_idx}-confs'], ani_results[name][f't{tautomer_idx}-energies'], rmsd_threshold=0.1)

    # remove most hydrogens
    reference_mol = _remove_hydrogens(copy.deepcopy(reference[0]))
    compare_mol = _remove_hydrogens(copy.deepcopy(mol))

    # find atom indices that are compared for RMSD
    sub_m = rdFMCS.FindMCS([reference_mol, compare_mol], bondCompare=Chem.rdFMCS.BondCompare.CompareOrder.CompareAny, maximizeBonds=False)
    mcsp = Chem.MolFromSmarts(sub_m.smartsString, False)

    # the order of the substructure lists are the same for both
    # substructure matches => substructure_idx_m1[i] = substructure_idx_m2[i]
    substructure_idx_reference = reference_mol.GetSubstructMatches(mcsp, uniquify=False)
    substructure_idx_compare = compare_mol.GetSubstructMatches(mcsp, uniquify=False)
    
    # generate rmsd matrix
    rmsd = np.zeros((reference_mol.GetNumConformers(), mol.GetNumConformers()),
                dtype=float)

    # save clusters
    got_hit = np.zeros(reference_mol.GetNumConformers(), dtype=int)

    # atom mapping
    from itertools import combinations
    for nr_of_mappings, (e1, e2) in enumerate(combinations(substructure_idx_reference+substructure_idx_compare, 2)):

        atom_mapping = [(a1, a2) for a1, a2 in zip(e1, e2)]
        # get rmsd matrix with a given set of atom mapping
        # update rmsd matrix whenever lower RMSD appears
        for i in range(len(reference_mol.GetConformers())):
            for j in range(len(compare_mol.GetConformers())):       
                
                proposed_rmsd = AllChem.AlignMol(reference_mol, compare_mol, i, j, atomMap=atom_mapping)
                # test if this is optimal atom mapping
                if nr_of_mappings == 0:
                    rmsd[i, j] = proposed_rmsd
                else:
                    rmsd[i, j] = min(rmsd[i, j], proposed_rmsd)

    for i in range(len(reference_mol.GetConformers())):
        for j in range(len(compare_mol.GetConformers())):       
            if  rmsd[i, j] <= 0.1:
                got_hit[i] += 1
                                          
    sns.heatmap(rmsd)
    plt.show()

    print(f'Nr of clusters: {len(got_hit)}')
    print(f'Nr of conformations part of one cluster: {sum(got_hit)}/{mol.GetNumConformers()}')
    print(f'Clusters present: {got_hit}')

    AllChem.AlignMolConformers(reference_mol)
    AllChem.AlignMolConformers(compare_mol)

    return compare_mol, minimum_traj, reference_mol, reference[1]


def _generate_conformer(coordinates):
    # generate a conformation object 
    new_conf = Chem.Conformer()
    for idx, c in enumerate(coordinates):
        point = Geometry.rdGeometry.Point3D()
        point.x = float(c[0].value_in_unit(unit.angstrom))
        point.y = float(c[1].value_in_unit(unit.angstrom))
        point.z = float(c[2].value_in_unit(unit.angstrom))
        new_conf.SetAtomPosition(idx, point)
    return new_conf

def get_data_filename():
    """
    In the source distribution, these files are in ``neutromeratio/data/*/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.
    """

    from pkg_resources import resource_filename
    fn = resource_filename('neutromeratio')

    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn


def setup_energy_function(name: str):

    data = pkg_resources.resource_stream(__name__, "data/exp_results.pickle")
    exp_results = pickle.load(data)

    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']
    
    #######################
    logger.debug(f"Experimental free energy difference: {exp_results[name]['energy']} kcal/mol")
    #######################
    
    ####################
    # Set up the system, set the restraints and read in the dcd files
    t_type, tautomers, flipped = generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles)
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    atoms = tautomer.hybrid_atoms

    # define the alchemical atoms
    alchemical_atoms = [tautomer.hybrid_hydrogen_idx_at_lambda_1, tautomer.hybrid_hydrogen_idx_at_lambda_0]
    model = LinearAlchemicalSingleTopologyANI(alchemical_atoms=alchemical_atoms)
    model = model.to(device)
    torch.set_num_threads(1)
    # extract hydrogen donor idx and hydrogen idx for from_mol

    # perform initial sampling
    energy_function = ANI1_force_and_energy(
        model=model,
        atoms=atoms,
        mol=None,
        per_atom_thresh=10.4 * unit.kilojoule_per_mole,
        adventure_mode=True
    )

    # add restraints
    for r in tautomer.ligand_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)
    for r in tautomer.hybrid_ligand_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)

    return energy_function, tautomer, flipped


def setup_mbar(name:str, data_path:str = "../data/", thinning:int = 50, max_snapshots_per_window:int = 200):
   
    def parse_lambda_from_dcd_filename(dcd_filename):
        """parsed the dcd filename

        Arguments:
            dcd_filename {str} -- how is the dcd file called?

        Returns:
            [float] -- lambda value
        """
        l = dcd_filename[:dcd_filename.find(f"_energy_in_vacuum")].split('_')
        lam = l[-3]
        return float(lam)
    
    data = pkg_resources.resource_stream(__name__, "data/exp_results.pickle")
    exp_results = pickle.load(data)

    if name in exclude_set_ANI + mols_with_charge + multiple_stereobonds:
        raise RuntimeError(f"{name} is part of the list of excluded molecules. Aborting")

    #######################
    energy_function, tautomer, flipped = setup_energy_function(name)
    # and lambda values in list
    dcds = glob(f"{data_path}/{name}/*.dcd")

    lambdas = []
    md_trajs = []
    energies = []

    # read in all the frames from the trajectories
    for dcd_filename in dcds:
        lam = parse_lambda_from_dcd_filename(dcd_filename)
        lambdas.append(lam)
        traj = md.load_dcd(dcd_filename, top=tautomer.hybrid_topology)[::thinning]
        logger.debug(f"Nr of frames in trajectory: {len(traj)}")
        md_trajs.append(traj)
        f = open(f"{data_path}/{name}/{name}_lambda_{lam:0.4f}_energy_in_vacuum.csv", 'r')
        energies.append(np.array([float(e) * kT for e in f][::thinning])) 
        f.close()

    assert (len(lambdas) > 5)
    assert(len(lambdas) == len(energies))
    assert(len(lambdas) == len(md_trajs))

    # calculate free energy in kT
    fec = FreeEnergyCalculator(ani_model=energy_function,
                                md_trajs=md_trajs,
                                potential_energy_trajs=energies,
                                lambdas=lambdas,
                                n_atoms=len(tautomer.hybrid_atoms),
                                max_snapshots_per_window=max_snapshots_per_window)

    fec.flipped = flipped

    return fec
