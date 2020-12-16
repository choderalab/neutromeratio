"""
Unit and regression test for the neutromeratio package.
"""

# Import package, test suite, and other packages as needed
import neutromeratio
from neutromeratio.constants import device
import pytest
import os, pickle


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Psi4 import fails on travis."
)
def test_psi4():
    from neutromeratio import qmpsi4

    exp_results = pickle.load(open("data/test_data/exp_results.pickle", "rb"))

    name = "molDWRow_298"

    t1_smiles = exp_results[name]["t1-smiles"]
    t2_smiles = exp_results[name]["t2-smiles"]

    # generate both rdkit mol
    (
        t_type,
        tautomers,
        flipped,
    ) = neutromeratio.utils.generate_tautomer_class_stereobond_aware(
        name, t1_smiles, t2_smiles, nr_of_conformations=5
    )
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    mol = tautomer.initial_state_mol

    psi4_mol = qmpsi4.mol2psi4(mol, 1)
    qmpsi4.optimize(psi4_mol)


def test_orca_input_generation():
    from neutromeratio import qmorca
    import rdkit

    name = "molDWRow_298"

    exp_results = pickle.load(open("data/test_data/exp_results.pickle", "rb"))
    t1_smiles = exp_results[name]["t1-smiles"]
    t2_smiles = exp_results[name]["t2-smiles"]

    qm_results = pickle.load(open("data/results/QM/qm_results.pickle", "rb"))
    mol = qm_results[name][t1_smiles]["vac"][0]

    orca_input = qmorca.generate_orca_script_for_solvation_free_energy(mol, 0)
    print(orca_input)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Orca not installed."
)
def test_running_orca():
    from neutromeratio import qmorca

    name = "molDWRow_298"

    exp_results = pickle.load(open("data/test_data/exp_results.pickle", "rb"))
    t1_smiles = exp_results[name]["t1-smiles"]
    t2_smiles = exp_results[name]["t2-smiles"]

    qm_results = pickle.load(open("data/results/QM/qm_results.pickle", "rb"))
    mol = qm_results[name][t1_smiles]["vac"][0]
    orca_input = qmorca.generate_orca_script_for_solvation_free_energy(mol, 0)
    f = open("tmp.inp", "w+")
    f.write(orca_input)
    f.close()
    out = qmorca.run_orca("tmp.inp")


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Orca not installed."
)
def test_solvate_orca():
    from neutromeratio import qmorca
    import re
    from simtk import unit
    from ..constants import hartree_to_kJ_mol

    name = "molDWRow_298"

    exp_results = pickle.load(open("data/test_data/exp_results.pickle", "rb"))
    t1_smiles = exp_results[name]["t1-smiles"]
    t2_smiles = exp_results[name]["t2-smiles"]

    qm_results = pickle.load(open("data/results/QM/qm_results.pickle", "rb"))
    mol = qm_results[name][t1_smiles]["vac"][0]
    orca_input = qmorca.generate_orca_script_for_solvation_free_energy(mol, 0)
    f = open("tmp.inp", "w+")
    f.write(orca_input)
    f.close()
    rc, output, err = qmorca.run_orca("tmp.inp")

    output_str = output.decode("UTF-8")

    try:
        # Total Energy after SMD CDS correction :
        found = re.search(
            "Total Energy after SMD CDS correction =\s*([-+]?\d*\.\d+|\d+)\s*Eh",
            output_str,
        ).group(1)
    except AttributeError:
        found = ""  # apply your error handling

    if not found:
        print(found)
        print(output_str)
        raise RuntimeError("Something not working. Aborting")

    print(found)
    print(output_str)

    E_in_solvent = float(found) * hartree_to_kJ_mol
    print(E_in_solvent)
    np.isclose(E_in_solvent, -907364.6683318849, rtol=1e-4)
