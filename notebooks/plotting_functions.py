from rdkit import Chem, Geometry
from rdkit.Chem import AllChem, rdFMCS, Draw
from simtk import unit
from neutromeratio.utils import generate_rdkit_mol
from neutromeratio.analysis import (
    calc_molecular_graph_entropy,
    calculate_weighted_energy,
    prune_conformers,
    compute_kl_divergence,
    bootstrap_rmse_r,
)
from neutromeratio.plotting import plot_correlation_analysis
from neutromeratio.constants import br_containing_mols
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle


def draw_mols(x_list: list, y_list: list, names: list):

    error_names = []
    error = []

    for x, y, name in zip(x_list, y_list, names):
        e = abs(float(x) - float(y))
        if e > 10:
            error_names.append(name)
            error.append(e)

    ps = rdFMCS.MCSParameters()
    ps.AtomCompareParameters.RingMatchesRingOnly = True
    ps.SetAtomTyper(rdFMCS.AtomCompare.CompareAny)
    for name in error_names:
        mol1, mol2 = (
            Chem.MolFromSmiles(exp_results[name]["t1-smiles"]),
            Chem.MolFromSmiles(exp_results[name]["t2-smiles"]),
        )
        mcs = Chem.MolFromSmarts(
            rdFMCS.FindMCS(
                [mol1, mol2],
                bondCompare=Chem.rdFMCS.BondCompare.CompareOrder.CompareAny,
            ).smartsString
        )
        AllChem.Compute2DCoords(mol1)
        match1 = mol1.GetSubstructMatch(mcs)
        match2 = mol2.GetSubstructMatch(mcs)
        coords = [mol1.GetConformer().GetAtomPosition(x) for x in match1]
        coords2D = [Geometry.Point2D(pt.x, pt.y) for pt in coords]
        coordDict = {}
        for i, coord in enumerate(coords2D):
            coordDict[match2[i]] = coord
        AllChem.Compute2DCoords(mol2, coordMap=coordDict)
    Draw.MolsToGridImage([mol1, mol2], subImgSize=(250, 250), molsPerRow=2)


def plot_two_distributions(x_list: list, y_list: list):
    fontsize = 27
    plt.figure(figsize=[8, 8], dpi=300)
    plt.subplot(211)
    sns.distplot(
        x_list, kde=True, rug=True, bins=15, label="$\Delta_{r}G_{solv}^{exp}$"
    )
    sns.distplot(
        y_list, kde=True, rug=True, bins=15, label="$\Delta_{r}G_{solv}^{calc}$"
    )

    plt.xlabel("$\Delta_{r}G_{solv}$ [kcal/mol]", fontsize=fontsize)
    plt.ylabel("Probability", fontsize=fontsize)
    plt.axvline(0, 0, 15, color="red")
    kl = compute_kl_divergence(np.array(x_list), np.array(y_list))
    rmse, mae, rho = bootstrap_rmse_r(np.array(x_list), np.array(y_list), 1000)
    plt.text(-13.0, 0.15, f"MAE$ = {mae}$", fontsize=fontsize)
    plt.text(-13.0, 0.13, f"RMSE$ = {rmse}$", fontsize=fontsize)
    plt.text(-13.0, 0.11, f"KL$ = {kl:.2f}$", fontsize=fontsize)
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    plt.legend(fontsize=fontsize)
    plt.ylabel("Probability")
    plt.yticks([])
    plt.tight_layout()
    plt.subplot(212)
    sns.distplot(
        np.array(x_list) - np.array(y_list),
        kde=True,
        rug=True,
        bins=15,
        label="$\Delta_{r}G_{solv}^{exp} - \Delta_{r}G_{solv}^{calc}$",
    )
    plt.xlabel(
        "$\Delta_{r}G_{solv}^{exp} - \Delta_{r}G_{solv}^{calc}$ [kcal/mol]",
        fontsize=fontsize,
    )
    plt.ylabel("Probability", fontsize=fontsize)
    plt.axvline(0, 0, 15, color="red")
    kl = compute_kl_divergence(np.array(x_list), np.array(y_list))
    rmse, mae, rho = bootstrap_rmse_r(np.array(x_list), np.array(y_list), 1000)
    plt.legend(fontsize=fontsize)
    plt.yticks([])

    plt.ylabel("Probability")
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)

    plt.tight_layout()
    plt.show()


def combine_conformations(list_of_mols_vac: list, list_of_mols_solv: list):
    molecule_with_multiple_confs_vac = Chem.Mol(list_of_mols_vac[0])
    molecule_with_multiple_confs_vac.RemoveAllConformers()

    molecule_with_multiple_confs_solv = Chem.Mol(list_of_mols_solv[0])
    molecule_with_multiple_confs_solv.RemoveAllConformers()

    molecules_vac = []
    molecules_solv = []

    vac_conf = dict(
        zip([int(m.GetProp("conf-id")) for m in list_of_mols_vac], list_of_mols_vac)
    )
    solv_conf = dict(
        zip([int(m.GetProp("conf-id")) for m in list_of_mols_solv], list_of_mols_solv)
    )

    for conf_id in vac_conf:
        if conf_id not in solv_conf.keys():
            # print(f"Skipping: {conf_id}")
            pass
        else:
            conf_vac = vac_conf[conf_id].GetConformer()
            molecule_with_multiple_confs_vac.AddConformer(conf_vac, assignId=True)
            molecules_vac.append(vac_conf[conf_id])

            conf_solv = solv_conf[conf_id].GetConformer()
            molecule_with_multiple_confs_solv.AddConformer(conf_solv, assignId=True)
            molecules_solv.append(solv_conf[conf_id])

    return (
        molecule_with_multiple_confs_vac,
        molecule_with_multiple_confs_solv,
        molecules_vac,
        molecules_solv,
    )


def get_min_free_energy_in_solution_method1(
    m_solv: list, correct_for_symmetry: bool = False
):

    # getting min values for vac/solv energy and free energy in vac
    if correct_for_symmetry:
        entropy_corr = calc_molecular_graph_entropy(m_solv[0])[0].value_in_unit(
            unit.kilocalorie_per_mole
        )
    else:
        entropy_corr = 0.0
    G_solv_min = min(
        [float(m.GetProp("G with S_rot corrected")) + entropy_corr for m in m_solv]
    )
    # getting min solvation free energty
    return G_solv_min * unit.kilocalorie_per_mole


def get_free_energy_in_solution_method1(
    solv_conf: Chem.Mol, m_solv: list, correct_for_symmetry: bool = False
):

    free_energy_in_solution = []
    if correct_for_symmetry:
        entropy_corr = calc_molecular_graph_entropy(solv_conf)[0].value_in_unit(
            unit.kilocalorie_per_mole
        )
    else:
        entropy_corr = 0.0

    # iterating over gas confs, adding energy in gas/solvation and free energy in gas for each conf

    for m in m_solv:
        G_solv = float(m.GetProp("G with S_rot corrected")) + entropy_corr
        free_energy_in_solution.append(G_solv * unit.kilocalorie_per_mole)

    mols, e_list = prune_conformers(
        solv_conf, free_energy_in_solution, rmsd_threshold=0.1
    )
    return e_list


def get_min_free_energy_in_solution_method2(
    m_vac: list, m_solv: list, solv_conf, correct_for_symmetry: bool = False
):

    if correct_for_symmetry:
        entropy_corr = calc_molecular_graph_entropy(solv_conf)[0].value_in_unit(
            unit.kilocalorie_per_mole
        )
    else:
        entropy_corr = 0.0

    # getting min values for vac/solv energy and free energy in vac
    dE_vac_min = min([float(m.GetProp("E_B3LYP_pVTZ")) for m in m_vac])
    dE_solv_min = min([float(m.GetProp("E_B3LYP_pVTZ")) for m in m_solv])
    G_vac_min = min(
        [float(m.GetProp("G with S_rot corrected")) + entropy_corr for m in m_vac]
    )
    # getting min solvation free energty
    return (G_vac_min + (dE_solv_min - dE_vac_min)) * unit.kilocalorie_per_mole


def get_free_energy_in_solution_method2(
    vac_conf: Chem.Mol,
    m_vac: list,
    m_solv: list,
    solv_conf,
    correct_for_symmetry: bool = False,
):

    if correct_for_symmetry:
        entropy_corr = calc_molecular_graph_entropy(solv_conf)[0].value_in_unit(
            unit.kilocalorie_per_mole
        )
    else:
        entropy_corr = 0.0

    free_energy_in_solution = []
    # iterating over gas confs, adding energy in gas/solvation and free energy in gas for each conf
    for m_v, m_s in zip(m_vac, m_solv):
        G_gas = float(m_v.GetProp("G with S_rot corrected")) + entropy_corr
        E_gas = float(m_v.GetProp("E_B3LYP_pVTZ"))
        E_solv = float(m_s.GetProp("E_B3LYP_pVTZ"))

        free_energy_in_solution.append(
            (G_gas + (E_solv - E_gas)) * unit.kilocalorie_per_mole
        )

    mols, e_list = prune_conformers(
        vac_conf, free_energy_in_solution, rmsd_threshold=0.1
    )
    return e_list


def get_min_free_energy_in_solution_method3(
    m_vac: list, m_solv: list, solv_conf, correct_for_symmetry: bool = False
):

    if correct_for_symmetry:
        entropy_corr = calc_molecular_graph_entropy(solv_conf)[0].value_in_unit(
            unit.kilocalorie_per_mole
        )
    else:
        entropy_corr = 0.0

    # getting min values for vac/solv energy and free energy in vac
    dE_vac_min = min([float(m.GetProp("E_B3LYP_631G_gas")) for m in m_vac])
    dE_solv_min = min([float(m.GetProp("E_B3LYP_631G_solv")) for m in m_solv])
    G_vac_min = min(
        [float(m.GetProp("G with S_rot corrected")) + entropy_corr for m in m_vac]
    )
    # getting min solvation free energty
    return (G_vac_min + (dE_solv_min - dE_vac_min)) * unit.kilocalorie_per_mole


def get_free_energy_in_solution_method3(
    vac_conf: Chem.Mol,
    m_vac: list,
    m_solv: list,
    solv_conf,
    correct_for_symmetry: bool = False,
):

    if correct_for_symmetry:
        entropy_corr = calc_molecular_graph_entropy(solv_conf)[0].value_in_unit(
            unit.kilocalorie_per_mole
        )
    else:
        entropy_corr = 0.0

    free_energy_in_solution = []
    # iterating over gas confs, adding energy in gas/solvation and free energy in gas for each conf
    for m_v, m_s in zip(m_vac, m_solv):
        G_gas = float(m_v.GetProp("G with S_rot corrected")) + entropy_corr
        E_gas = float(m_v.GetProp("E_B3LYP_631G_gas"))
        E_solv = float(m_s.GetProp("E_B3LYP_631G_solv"))

        free_energy_in_solution.append(
            (G_gas + (E_solv - E_gas)) * unit.kilocalorie_per_mole
        )

    mols, e_list = prune_conformers(
        vac_conf, free_energy_in_solution, rmsd_threshold=0.1
    )
    return e_list


def calc_rot_entropy(deg: int):
    return -1 * gas_constant * np.log(deg)


def combine_y_and_exp(method, combining_method, b3lyp_results):
    exp_results = pickle.load(open("../data/results/exp_results.pickle", "rb"))
    x_list = []
    y_list = []
    names = []
    for name in b3lyp_results:
        if method == "method3" and name in br_containing_mols:
            # skip bromid containing molecules
            print(name)
            continue

        x = exp_results[name]["energy [kcal/mol]"]
        y = (
            b3lyp_results[name][f"t2-free-energy-solv-{method}-{combining_method}"]
            - b3lyp_results[name][f"t1-free-energy-solv-{method}-{combining_method}"]
        )
        x = x.value_in_unit(unit.kilocalorie_per_mole)
        y = y.value_in_unit(unit.kilocalorie_per_mole)
        if x < 0.0:
            x *= -1.0
            y *= -1.0
        x_list.append(x)
        y_list.append(y)
        names.append(name)

    if len(x_list) != len(y_list):
        print(method)
        print(combining_method)
        raise RuntimeError()

    error = []
    for i in range(len(x_list)):
        error.append(x_list[i] - y_list[i])
    outlier = []
    for i in range(len(x_list)):
        o = abs(x_list[i] - y_list[i])
        if o > 10:
            outlier.append(names[i])
    return x_list, y_list, names, error, outlier


def plot_single_dist(x_list, y_list, label):
    fontsize = 27
    plt.figure(figsize=[8, 8], dpi=300)
    plt.ylabel("Probability", fontsize=fontsize)

    from neutromeratio.analysis import compute_kl_divergence, bootstrap_rmse_r

    sns.distplot(
        np.array(x_list) - np.array(y_list),
        kde=True,
        rug=True,
        bins=15,
        label="Boltzmann weighting - Minimum",
    )
    rmse, mae, rho = bootstrap_rmse_r(np.array(x_list), np.array(y_list), 1000)
    kl = compute_kl_divergence(np.array(x_list), np.array(y_list))
    # plt.text(-1.5, 1.5, f"RMSE$ = {rmse}$", fontsize=fontsize)
    # plt.text(-1.5, 1.35, f"MAE$ = {mae}$", fontsize=fontsize)
    # plt.text(8.0, 0.04, f"KL$ = {kl:.2f}$", fontsize=fontsize)
    plt.xlabel(label, fontsize=fontsize)
    plt.axvline(0, 0, 15, color="red")

    # plt.legend(fontsize=fontsize)
    plt.yticks([])
    plt.show()


def combine_x_and_y(method: str, b3lyp_results: dict):
    x_list = []
    y_list = []
    names = []
    for name in b3lyp_results:
        if method == "method3" and name in br_containing_mols:
            # skip bromid containing molecules
            continue

        x = (
            b3lyp_results[name][f"t2-free-energy-solv-{method}-mm"]
            - b3lyp_results[name][f"t1-free-energy-solv-{method}-mm"]
        )
        y = (
            b3lyp_results[name][f"t2-free-energy-solv-{method}-min"]
            - b3lyp_results[name][f"t1-free-energy-solv-{method}-min"]
        )
        x = x.value_in_unit(unit.kilocalorie_per_mole)
        y = y.value_in_unit(unit.kilocalorie_per_mole)
        if x < 0.0:
            x *= -1.0
            y *= -1.0
        x_list.append(x)
        y_list.append(y)
        names.append(name)

    if len(x_list) != len(y_list):
        print(method)
        print(combining_method)
        raise RuntimeError()

    error = []
    for i in range(len(x_list)):
        error.append(x_list[i] - y_list[i])

    return x_list, y_list, names, error


def plot_diff(x_list: list, y_list: list):
    fontsize = 27
    from neutromeratio.analysis import compute_kl_divergence, bootstrap_rmse_r

    plt.figure(figsize=[8, 8], dpi=300)

    plt.subplot(211)
    sns.distplot(x_list, kde=True, rug=True, bins=15, label="$\Delta G_{solv}^{exp}$")
    sns.distplot(y_list, kde=True, rug=True, bins=15, label="$\Delta G_{solv}^{calc}$")

    plt.xlabel("$\Delta G_{solv}$ [kcal/mol]", fontsize=fontsize)
    plt.ylabel("Probability", fontsize=fontsize)
    plt.axvline(0, 0, 15, color="red")
    kl = compute_kl_divergence(np.array(x_list), np.array(y_list))
    rmse, mae, rho = bootstrap_rmse_r(np.array(x_list), np.array(y_list), 1000)
    plt.text(-13.0, 0.15, f"MAE$ = {mae}$", fontsize=fontsize)
    plt.text(-13.0, 0.13, f"RMSE$ = {rmse}$", fontsize=fontsize)
    plt.text(-13.0, 0.11, f"KL$ = {kl:.2f}$", fontsize=fontsize)
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    plt.legend(fontsize=fontsize)
    plt.ylabel("Probability")
    plt.yticks([])
    plt.tight_layout()
    plt.subplot(212)
    sns.distplot(
        np.array(x_list) - np.array(y_list),
        kde=True,
        rug=True,
        bins=15,
        label="$\Delta G_{solv}^{exp} - \Delta G_{solv}^{calc}$",
    )
    plt.xlabel(
        "$\Delta G_{solv}^{exp} - \Delta G_{solv}^{calc}$ [kcal/mol]", fontsize=fontsize
    )
    plt.ylabel("Probability", fontsize=fontsize)
    plt.axvline(0, 0, 15, color="red")
    kl = compute_kl_divergence(np.array(x_list), np.array(y_list))
    rmse, mae, rho = bootstrap_rmse_r(np.array(x_list), np.array(y_list), 1000)
    plt.legend(fontsize=fontsize)
    plt.yticks([])

    plt.ylabel("Probability")
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)

    plt.tight_layout()

    plt.show()