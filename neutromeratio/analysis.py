import torch
import copy
import logging
import pickle
from glob import glob

import matplotlib.pyplot as plt
import mdtraj as md
import networkx as nx
import numpy as np
import pandas as pd
import pkg_resources
import scipy.stats as scs
import seaborn as sns
import torchani
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem, rdFMCS
from scipy.special import logsumexp
from simtk import unit

from neutromeratio.ani import (
    ANI,
    AlchemicalANI1ccx,
    AlchemicalANI1x,
    AlchemicalANI2x,
    ANI1_force_and_energy,
)
from neutromeratio.constants import (
    device,
    exclude_set_ANI,
    gas_constant,
    kT,
    mols_with_charge,
    multiple_stereobonds,
    num_threads,
    temperature,
)
from neutromeratio.parameter_gradients import FreeEnergyCalculator
from neutromeratio.tautomers import Tautomer
from neutromeratio.utils import (
    generate_new_tautomer_pair,
    generate_tautomer_class_stereobond_aware,
)

logger = logging.getLogger(__name__)


def _remove_hydrogens(m: Chem.Mol):
    """Removing all hydrogens from the molecule with a few exceptions"""
    #########################################
    # search for important hydrogens
    #########################################
    keep_hydrogens = []
    logger.debug("search for patterns ...")
    # test for primary alcohol
    patt = Chem.MolFromSmarts("[OX2H]")
    if m.HasSubstructMatch(patt):
        logger.debug("found primary alcohol")
        l = m.GetSubstructMatch(patt)
        keep_hydrogens.extend(l)

    # test for imine
    patt = Chem.MolFromSmarts("[CX3]=[NH]")
    if m.HasSubstructMatch(patt):
        logger.debug("found imine")
        l = m.GetSubstructMatch(patt)
        keep_hydrogens.extend(l)

    # test for primary amine
    patt = Chem.MolFromSmarts("[NX3;H2]")
    if m.HasSubstructMatch(patt):
        logger.debug("found primary amine")
        l = m.GetSubstructMatch(patt)
        keep_hydrogens.extend(l)

    # test for secondary amine
    patt = Chem.MolFromSmarts("[NX3H]")
    if m.HasSubstructMatch(patt):
        logger.debug("found secondary amine")
        l = m.GetSubstructMatch(patt)
        keep_hydrogens.extend(l)

    # test for cyanamide
    patt = Chem.MolFromSmarts("[NX3][CX2]#[NX1]")
    if m.HasSubstructMatch(patt):
        logger.debug("found cyanamide")
        l = m.GetSubstructMatch(patt)
        keep_hydrogens.extend(l)

    # test for thiol
    patt = Chem.MolFromSmarts("[#16X2H]")
    if m.HasSubstructMatch(patt):
        logger.debug("found thiol")
        l = m.GetSubstructMatch(patt)
        keep_hydrogens.extend(l)

    # unfortunatelly, RemoveHs() does not retain marked hydrogens
    # therefore, mutating important Hs to Li and then remutating them
    # to Hs
    if keep_hydrogens:
        for idx in keep_hydrogens:
            atom = m.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == "H":
                    hydrogen = m.GetAtomWithIdx(neighbor.GetIdx())
                    hydrogen.SetAtomicNum(3)

        m = Chem.RemoveHs(m)
        for atom in m.GetAtoms():
            if atom.GetSymbol() == "Li":
                hydrogen = m.GetAtomWithIdx(atom.GetIdx())
                hydrogen.SetAtomicNum(1)
    else:
        m = Chem.RemoveHs(m)

    return m


def prune_conformers(
    mol: Chem.Mol, energies: list, rmsd_threshold: float
) -> (Chem.Mol, list):
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
    sort = np.argsort(
        [x.value_in_unit(unit.kilocalorie_per_mole) for x in energies]
    )  # sort by increasing energy
    keep: List[int] = []  # always keep lowest-energy conformer
    discard: List[int] = []
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
    logger.debug(f"keep: {keep}")
    for i in keep:
        logger.debug(i)
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
    rmsd = np.zeros((mol.GetNumConformers(), mol.GetNumConformers()), dtype=float)

    mol = _remove_hydrogens(copy.deepcopy(mol))
    for i, ref_conf in enumerate(mol.GetConformers()):
        for j, fit_conf in enumerate(mol.GetConformers()):
            if i >= j:
                continue
            rmsd[i, j] = AllChem.GetBestRMS(
                mol, mol, ref_conf.GetId(), fit_conf.GetId()
            )
            rmsd[j, i] = rmsd[i, j]
    return rmsd


def calculate_weighted_energy(e_list):
    # G = -RT ln Σ exp(-G/RT)
    e_bw = (
        (-1)
        * gas_constant
        * temperature
        * (logsumexp([(-1) * (e) / (gas_constant * temperature) for e in e_list]))
    )
    return e_bw


def mol_to_nx(mol: Chem.Mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            atomic_num=atom.GetAtomicNum(),
            formal_charge=atom.GetFormalCharge(),
            chiral_tag=atom.GetChiralTag(),
            hybridization=atom.GetHybridization(),
            num_explicit_hs=atom.GetNumExplicitHs(),
            is_aromatic=atom.GetIsAromatic(),
        )

    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType()
        )
    return G


def small_molecule_degeneracy(graph):
    # Now iterate over graph isomorphisms
    def equ_node(n1, n2):
        if (
            int(n1["atomic_num"]) == int(n2["atomic_num"])
            and n1["hybridization"] == n2["hybridization"]
        ):
            return True
        else:
            return False

    def equ_edge(n1, n2):
        if int(n1["bond_type"]) == int(n2["bond_type"]):
            return True
        else:
            return False

    from networkx.algorithms.isomorphism import GraphMatcher

    graph_matcher = GraphMatcher(graph, graph, node_match=equ_node, edge_match=equ_edge)
    degeneracy = sum([1 for isomorphism in graph_matcher.match()])
    return calc_rot_entropy(degeneracy), degeneracy


def calc_rot_entropy(deg: int):
    return -1 * gas_constant * temperature * np.log(deg)


def calc_molecular_graph_entropy(mol):
    return small_molecule_degeneracy(mol_to_nx(mol))


def compare_confomer_generator_and_trajectory_minimum_structures(
    results_path: str, name: str, base: str, tautomer_idx: int, thinning: int = 100
):
    assert tautomer_idx == 1 or tautomer_idx == 2

    ani_results = pickle.load(open(f"{results_path}/ani_mm_results.pickle", "rb"))
    exp_results = pickle.load(open(f"{results_path}/exp_results.pickle", "rb"))

    # generate the tautomer object
    t1_smiles = exp_results[name]["t1-smiles"]
    t2_smiles = exp_results[name]["t2-smiles"]
    t_type, tautomers, flipped = generate_tautomer_class_stereobond_aware(
        name, t1_smiles, t2_smiles, nr_of_conformations=1, enforceChirality=True
    )

    tautomer = tautomers[0]
    print(f"Flipped: {flipped}")
    tautomer.perform_tautomer_transformation()

    tautomer_mol = prune_conformers(
        ani_results[name]["t1-confs"],
        ani_results[name]["t1-energies"],
        rmsd_threshold=0.1,
    )
    print(len(tautomer_mol[1]))

    traj_path = (
        f"{base}/{name}/{name}_lambda_{tautomer_idx-1}.0000_kappa_0.0000_in_vacuum.dcd"
    )
    pdb_path = f"{base}/{name}/{name}_0.pdb"

    # load trajectory, remove dummy atom
    traj = md.load(traj_path, top=pdb_path)
    atom_idx = [a.index for a in traj.topology.atoms]
    if (tautomer_idx - 1) == 1:
        atom_idx.remove(int(tautomer.hybrid_hydrogen_idx_at_lambda_0))
    else:
        atom_idx.remove(int(tautomer.hybrid_hydrogen_idx_at_lambda_1))

    traj = traj.atom_slice(atom_indices=atom_idx)

    # save pdb without dummy atom
    tautomer_pdb = f"{base}/{name}/{name}_without_dummy_{tautomer_idx}.pdb"
    traj[0].save_pdb(tautomer_pdb)

    # generate rdkit mol object with the same atom indizes as the trajectory but without the dummy atom
    mol = Chem.MolFromPDBFile(tautomer_pdb, removeHs=False)
    # remove conf of pdb
    mol.RemoveAllConformers()

    # generate energy function, use atom symbols of rdkti mol
    from .ani import ANI1_force_and_energy, ANI1ccx

    model = ANI1ccx()
    energy_function = ANI1_force_and_energy(
        model=model, atoms=[a.GetSymbol() for a in mol.GetAtoms()], mol=None
    )

    # take every 100th conformation and minimize it using ANI1
    minimized_traj = []  # store min conformations in here

    for idx, conf in enumerate(traj[::thinning]):

        print(f"{idx}/{len(traj[::thinning])}")
        c = (conf.xyz[0]) * unit.nanometer
        min_conf = energy_function.minimize(c)[
            0
        ]  # only real atoms, therefor lambda not needed
        minimized_traj.append(min_conf)
        new_conf = _generate_conformer(min_conf)
        # add the conformation to the rdkit mol object
        mol.AddConformer(new_conf, assignId=True)

    # generate mdtraj object with minimized confs
    minimum_traj = md.Trajectory(
        np.array([v.value_in_unit(unit.nanometer) for v in minimized_traj]),
        traj.topology,
    )

    # generate reference_mol
    reference = prune_conformers(
        ani_results[name][f"t{tautomer_idx}-confs"],
        ani_results[name][f"t{tautomer_idx}-energies"],
        rmsd_threshold=0.1,
    )

    # remove most hydrogens
    reference_mol = _remove_hydrogens(copy.deepcopy(reference[0]))
    compare_mol = _remove_hydrogens(copy.deepcopy(mol))

    # find atom indices that are compared for RMSD
    sub_m = rdFMCS.FindMCS(
        [reference_mol, compare_mol],
        bondCompare=Chem.rdFMCS.BondCompare.CompareOrder.CompareAny,
        maximizeBonds=False,
    )
    mcsp = Chem.MolFromSmarts(sub_m.smartsString, False)

    # the order of the substructure lists are the same for both
    # substructure matches => substructure_idx_m1[i] = substructure_idx_m2[i]
    substructure_idx_reference = reference_mol.GetSubstructMatches(mcsp, uniquify=False)
    substructure_idx_compare = compare_mol.GetSubstructMatches(mcsp, uniquify=False)

    # generate rmsd matrix
    rmsd = np.zeros(
        (reference_mol.GetNumConformers(), mol.GetNumConformers()), dtype=float
    )

    # save clusters
    got_hit = np.zeros(reference_mol.GetNumConformers(), dtype=int)

    # atom mapping
    from itertools import combinations

    for nr_of_mappings, (e1, e2) in enumerate(
        combinations(substructure_idx_reference + substructure_idx_compare, 2)
    ):

        atom_mapping = [(a1, a2) for a1, a2 in zip(e1, e2)]
        # get rmsd matrix with a given set of atom mapping
        # update rmsd matrix whenever lower RMSD appears
        for i in range(len(reference_mol.GetConformers())):
            for j in range(len(compare_mol.GetConformers())):

                proposed_rmsd = AllChem.AlignMol(
                    reference_mol, compare_mol, i, j, atomMap=atom_mapping
                )
                # test if this is optimal atom mapping
                if nr_of_mappings == 0:
                    rmsd[i, j] = proposed_rmsd
                else:
                    rmsd[i, j] = min(rmsd[i, j], proposed_rmsd)

    for i in range(len(reference_mol.GetConformers())):
        for j in range(len(compare_mol.GetConformers())):
            if rmsd[i, j] <= 0.1:
                got_hit[i] += 1

    sns.heatmap(rmsd)
    plt.show()

    print(f"Nr of clusters: {len(got_hit)}")
    print(
        f"Nr of conformations part of one cluster: {sum(got_hit)}/{mol.GetNumConformers()}"
    )
    print(f"Clusters present: {got_hit}")

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

    fn = resource_filename("neutromeratio")

    if not os.path.exists(fn):
        raise ValueError(
            "Sorry! %s does not exist. If you just added it, you'll have to re-install"
            % fn
        )

    return fn


def _get_exp_results():
    data = pkg_resources.resource_stream(__name__, "data/exp_results.pickle")
    exp_results = pickle.load(data)
    return exp_results


def setup_alchemical_system_and_energy_function(
    name: str,
    env: str,
    ANImodel: ANI,
    checkpoint_file: str = "",
    base_path: str = None,
    diameter: int = -1,
):

    import os

    if not (
        issubclass(ANImodel, (AlchemicalANI2x, AlchemicalANI1ccx, AlchemicalANI1x))
    ):
        raise RuntimeError("Only Alchemical ANI objects allowed! Aborting.")

    exp_results = _get_exp_results()
    t1_smiles = exp_results[name]["t1-smiles"]
    t2_smiles = exp_results[name]["t2-smiles"]

    #######################
    logger.debug(
        f"Experimental free energy difference: {exp_results[name]['energy']} kcal/mol"
    )
    #######################

    ####################
    # Set up the system, set the restraints
    t_type, tautomers, flipped = generate_tautomer_class_stereobond_aware(
        name, t1_smiles, t2_smiles
    )
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    # if base_path is defined write out the topology
    if base_path:
        base_path = os.path.abspath(base_path)
        logger.debug(base_path)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

    if env == "droplet":
        if diameter == -1:
            raise RuntimeError("Droplet is not specified. Aborting.")
        # for droplet topology is written in every case
        m = tautomer.add_droplet(
            tautomer.hybrid_topology,
            tautomer.get_hybrid_coordinates(),
            diameter=diameter * unit.angstrom,
            restrain_hydrogen_bonds=True,
            restrain_hydrogen_angles=False,
            top_file=f"{base_path}/{name}_in_droplet.pdb",
        )
    else:
        if base_path:
            # for vacuum only if base_path is defined
            pdb_filepath = f"{base_path}/{name}.pdb"
            try:
                traj = md.load(pdb_filepath)
            except OSError:
                coordinates = tautomer.get_hybrid_coordinates()
                traj = md.Trajectory(
                    coordinates.value_in_unit(unit.nanometer), tautomer.hybrid_topology
                )
                traj.save_pdb(pdb_filepath)
            tautomer.set_hybrid_coordinates(traj.xyz[0] * unit.nanometer)

    # define the alchemical atoms
    alchemical_atoms = [
        tautomer.hybrid_hydrogen_idx_at_lambda_1,
        tautomer.hybrid_hydrogen_idx_at_lambda_0,
    ]

    model = ANImodel(alchemical_atoms=alchemical_atoms).to(device)
    # if specified, load nn parameters
    if checkpoint_file:
        logger.debug("Loading nn parameters ...")
        model.load_nn_parameters(checkpoint_file)

    # setup energy function
    if env == "vacuum":
        energy_function = ANI1_force_and_energy(
            model=model,
            atoms=tautomer.hybrid_atoms,
            mol=None,
        )
    else:
        energy_function = ANI1_force_and_energy(
            model=model,
            atoms=tautomer.ligand_in_water_atoms,
            mol=None,
        )

    # add restraints
    for r in tautomer.ligand_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)
    for r in tautomer.hybrid_ligand_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)

    if env == "droplet":
        tautomer.add_COM_for_hybrid_ligand(
            np.array([diameter / 2, diameter / 2, diameter / 2]) * unit.angstrom
        )
        for r in tautomer.solvent_restraints:
            energy_function.add_restraint_to_lambda_protocol(r)
        for r in tautomer.com_restraints:
            energy_function.add_restraint_to_lambda_protocol(r)

    return energy_function, tautomer, flipped


def setup_new_alchemical_system_and_energy_function(
    name: str,
    t1_smiles: str,
    t2_smiles: str,
    env: str,
    ANImodel: ANI,
    base_path: str = None,
    diameter: int = -1,
    checkpoint_file: str = "",
):

    import os

    if not (
        issubclass(ANImodel, (AlchemicalANI2x, AlchemicalANI1ccx, AlchemicalANI1x))
    ):
        raise RuntimeError("Only Alchemical ANI objects allowed! Aborting.")

    #######################

    ####################
    # Set up the system, set the restraints
    tautomer = generate_new_tautomer_pair(name, t1_smiles, t2_smiles)
    tautomer.perform_tautomer_transformation()

    # if base_path is defined write out the topology
    if base_path:
        base_path = os.path.abspath(base_path)
        logger.debug(base_path)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

    if env == "droplet":
        if diameter == -1:
            raise RuntimeError("Droplet is not specified. Aborting.")
        # for droplet topology is written in every case
        m = tautomer.add_droplet(
            tautomer.hybrid_topology,
            tautomer.get_hybrid_coordinates(),
            diameter=diameter * unit.angstrom,
            restrain_hydrogen_bonds=True,
            restrain_hydrogen_angles=False,
            top_file=f"{base_path}/{name}_in_droplet.pdb",
        )
    else:
        if base_path:
            # for vacuum only if base_path is defined
            pdb_filepath = f"{base_path}/{name}.pdb"
            try:
                traj = md.load(pdb_filepath)
            except OSError:
                coordinates = tautomer.get_hybrid_coordinates()
                traj = md.Trajectory(
                    coordinates.value_in_unit(unit.nanometer), tautomer.hybrid_topology
                )
                traj.save_pdb(pdb_filepath)
            tautomer.set_hybrid_coordinates(traj.xyz[0] * unit.nanometer)

    # define the alchemical atoms
    alchemical_atoms = [
        tautomer.hybrid_hydrogen_idx_at_lambda_1,
        tautomer.hybrid_hydrogen_idx_at_lambda_0,
    ]

    model = ANImodel(alchemical_atoms=alchemical_atoms).to(device)
    # if specified, load nn parameters
    if checkpoint_file:
        logger.debug("Loading nn parameters ...")
        model.load_nn_parameters(checkpoint_file)

    # setup energy function
    if env == "vacuum":
        energy_function = ANI1_force_and_energy(
            model=model,
            atoms=tautomer.hybrid_atoms,
            mol=None,
        )
    else:
        energy_function = ANI1_force_and_energy(
            model=model,
            atoms=tautomer.ligand_in_water_atoms,
            mol=None,
        )

    # add restraints
    for r in tautomer.ligand_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)
    for r in tautomer.hybrid_ligand_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)

    if env == "droplet":
        tautomer.add_COM_for_hybrid_ligand(
            np.array([diameter / 2, diameter / 2, diameter / 2]) * unit.angstrom
        )
        for r in tautomer.solvent_restraints:
            energy_function.add_restraint_to_lambda_protocol(r)
        for r in tautomer.com_restraints:
            energy_function.add_restraint_to_lambda_protocol(r)

    return energy_function, tautomer


def _error(x: np.ndarray, y: np.ndarray):
    """ Simple error """
    return x - y


def array_rmse(x: np.ndarray, y: np.ndarray) -> float:
    """Returns the root mean squared error between two arrays."""
    return np.sqrt(((x - y) ** 2).mean())


def array_mae(x: np.ndarray, y: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(x, y)))


from typing import Tuple


def bootstrap_tautomer_exp_predict_results(
    exp_original: list, pred_original: list
) -> Tuple[list, list]:
    """Perform empirical bootstrap over rows for correlation analysis."""
    size = len(exp_original)
    rows = np.random.choice(np.arange(size), size=size)
    return (
        np.array([exp_original[i] for i in rows]),
        np.array([pred_original[i] for i in rows]),
    )


def bootstrap_rmse_r(exp_original: np.array, pred_original: np.array, nsamples: int):
    """Perform a bootstrap correlation analysis for a dataframe

    Parameters
    ----------
    exp_original - the original np.array with experimental data.
    pred_original - the original np.array with preidcted data
    nsamples - number of bootstrap samples to draw
    """

    rmse_list = list()
    rs_list = list()
    mae_list = list()
    for _ in range(nsamples):
        exp, pred = bootstrap_tautomer_exp_predict_results(exp_original, pred_original)
        mae_list.append(array_mae(exp, pred))
        rmse_list.append(array_rmse(exp, pred))
        rs_list.append(scs.pearsonr(exp, pred)[0])

    rmse_array = np.asarray(rmse_list)
    rs_array = np.asarray(rs_list)
    mae_array = np.asarray(mae_list)

    rmse = array_rmse(exp_original, pred_original)
    rs = scs.pearsonr(exp_original, pred_original)[0]
    mae = array_mae(exp_original, pred_original)

    return (
        BootstrapDistribution(rmse, rmse_array),
        BootstrapDistribution(mae, mae_array),
        BootstrapDistribution(rs, rs_array),
    )


class BootstrapDistribution:
    """Represents a bootstrap distribution, and allows calculation of useful statistics."""

    def __init__(self, sample_estimate: float, bootstrap_estimates: np.array) -> None:
        """
        Parameters
        ----------
        sample_estimate - estimated value from original sample
        bootstrap_estimates - estimated values from bootstrap samples
        """
        self._sample = sample_estimate
        self._bootstrap = bootstrap_estimates
        # approximation of δ = X - μ
        # denoted as δ* = X* - X
        self._delta = bootstrap_estimates - sample_estimate
        self._significance: int = 16
        return

    def __float__(self):
        return float(self._sample)

    def empirical_bootstrap_confidence_intervals(self, percent: float):
        """Return % confidence intervals.

        Uses the approximation that the variation around the center of the
        bootstrap distribution is the same as the variation around the
        center of the true distribution. E.g. not sensitive to a bootstrap
        distribution is restraint_biased in the median.

        Note
        ----
        Estimates are rounded to the nearest significant digit.
        """
        if not 0.0 < percent < 100.0:
            raise ValueError("Percentage should be between 0.0 and 100.0")
        elif 0.0 < percent < 1.0:
            raise ValueError("Received a fraction but expected a percentile.")

        a = (100.0 - percent) / 2
        upper = np.percentile(self._delta, a)
        lower = np.percentile(self._delta, 100 - a)
        return self._sig_figures(
            self._sample, self._sample - lower, self._sample - upper
        )

    def bootstrap_percentiles(self, percent: float):
        """Return percentiles of the bootstrap distribution.

        This assumes that the bootstrap distribution is similar to the real
        distribution. This breaks down when the median of the bootstrap
        distribution is very different from the sample/true variable median.

        Note
        ----
        Estimates are rounded to the nearest significant digit.
        """
        if not 0.0 < percent < 100.0:
            raise ValueError("Percentage should be between 0.0 and 100.0")
        elif 0.0 < percent < 1.0:
            raise UserWarning("Received a fraction but expected a percentage.")

        a = (100.0 - percent) / 2
        lower = np.percentile(self._bootstrap, a)
        upper = np.percentile(self._bootstrap, 100 - a)
        return self._sig_figures(self._sample, lower, upper)

    def standard_error(self) -> float:
        """Return the standard error for the bootstrap estimate."""
        # Is calculated by taking the standard deviation of the bootstrap distribution
        return self._bootstrap.std()

    def _sig_figures(self, mean: float, lower: float, upper: float, max_sig: int = 16):
        """Find the lowest number of significant figures that distinguishes
        the mean from the lower and upper bound.
        """
        i = 16
        for i in range(1, max_sig):
            if (round(mean, i) != round(lower, i)) and (
                round(mean, i) != round(upper, i)
            ):
                break
        self._significance = i
        return round(mean, i), round(lower, i), round(upper, i)

    def __repr__(self) -> str:
        try:
            return "{0:.{3}f}; [{1:.{3}f}, {2:.{3}f}]".format(
                *self.bootstrap_percentiles(95), self._significance
            )
        except:
            return str(self.__dict__)

    def _round_float(self, flt):
        return (flt * pow(10, self._significance)) / pow(10, self._significance)

    def to_tuple(self, percent=95):
        """Return the mean, lower and upper percentiles rounded to significant digits"""
        mean, lower, upper = self.bootstrap_percentiles(percent)
        mean = self._round_float(mean)
        lower = self._round_float(lower)
        upper = self._round_float(upper)
        return mean, lower, upper


# taken from here:
# https://medium.com/datalab-log/measuring-the-statistical-similarity-between-two-samples-using-jensen-shannon-and-kullback-leibler-8d05af514b15
def compute_probs(data, n=10):
    h, e = np.histogram(data, n)
    p = h / data.shape[0]
    return e, p


def support_intersection(p, q):
    return list(filter(lambda x: (x[0] != 0) & (x[1] != 0), list(zip(p, q))))


def get_probs(list_of_tuples):
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q


def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))


def js_divergence(p, q):
    m = (1.0 / 2.0) * (p + q)
    return (1.0 / 2.0) * kl_divergence(p, m) + (1.0 / 2.0) * kl_divergence(q, m)


def compute_kl_divergence(train_sample, test_sample, n_bins=10):
    """
    Computes the KL Divergence using the support intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)
    return kl_divergence(p, q)
