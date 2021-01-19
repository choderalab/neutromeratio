"""
Unit and regression test for the neutromeratio package.
"""

# Import package, test suite, and other packages as needed
import neutromeratio
import pytest
import sys
import os
import pickle
import torch
from simtk import unit
import numpy as np
import mdtraj as md
from neutromeratio.constants import device
from openmmtools.utils import is_quantity_close
from neutromeratio.constants import device


def test_equ():
    assert 1.0 == 1.0


def test_neutromeratio_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "neutromeratio" in sys.modules


def test_equilibrium():
    # test the langevin dynamics with different neural net potentials
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..equilibrium import LangevinDynamics
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x
    from ..constants import _get_names

    # name of the system
    name = "molDWRow_298"
    # number of steps
    n_steps = 100

    for model_name, model in zip(
        ["AlchemicalANI2x", "AlchemicalANI1ccx"], [AlchemicalANI2x, AlchemicalANI1ccx]
    ):
        (energy_function, tautomer, _,) = setup_alchemical_system_and_energy_function(
            name=name, ANImodel=model, env="vacuum", base_path="pdbs"
        )

        x0 = tautomer.get_hybrid_coordinates()  # format [1][K][3] * unit
        x0, _ = energy_function.minimize(x0)

        # lambda=1.0
        energy_and_force = lambda x: energy_function.calculate_force(x, 1.0)

        langevin = LangevinDynamics(
            atoms=tautomer.hybrid_atoms,
            energy_and_force=energy_and_force,
        )

        equilibrium_samples, _, _ = langevin.run_dynamics(
            x0, n_steps=n_steps, stepsize=1.0 * unit.femtosecond, progress_bar=True
        )

        equilibrium_samples = [
            x[0].value_in_unit(unit.nanometer) for x in equilibrium_samples
        ]
        traj = md.Trajectory(equilibrium_samples, tautomer.hybrid_topology)
        traj.save(f"test_{model_name}.dcd", force_overwrite=True)
        traj[0].save("test.pdb")

        # lambda=0.0
        energy_and_force = lambda x: energy_function.calculate_force(x, 0.0)

        langevin = LangevinDynamics(
            atoms=tautomer.hybrid_atoms, energy_and_force=energy_and_force
        )

        equilibrium_samples, _, _ = langevin.run_dynamics(
            x0, n_steps=n_steps, stepsize=1.0 * unit.femtosecond, progress_bar=True
        )


def test_calculate_single_energy():
    import torch
    import torchani

    torch.set_num_threads(1)
    n_snapshots, n_atoms = 1, 50

    device = torch.device("cpu")
    model = torchani.models.ANI1ccx(periodic_table_index=True).to(device)

    element_index = {"C": 6, "N": 7, "O": 8, "H": 1}
    species = [element_index[e] for e in "CH" * n_atoms][:n_atoms]
    species_tensor = torch.tensor([species] * n_snapshots, device=device)

    # condition 1: computing n_snapshots energies
    print("computing once")
    coordinates = torch.tensor(
        torch.randn((n_snapshots, n_atoms, 3)),
        requires_grad=True,
        device=device,
        dtype=torch.float32,
    )
    energy = model((species_tensor, coordinates)).energies

    print(energy)


def test_mining_minima():
    # name of the system
    name = "molDWRow_298"
    # number of steps
    exp_results = pickle.load(open("data/test_data/exp_results.pickle", "rb"))
    torch.set_num_threads(1)

    t1_smiles = exp_results[name]["t1-smiles"]
    t2_smiles = exp_results[name]["t2-smiles"]

    (_, tautomers, _,) = neutromeratio.utils.generate_tautomer_class_stereobond_aware(
        name, t1_smiles, t2_smiles
    )
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    (
        confs_traj,
        e,
        minimum_energies,
        all_energies,
        all_conformations,
    ) = tautomer.generate_mining_minima_structures()


def test_generating_droplet():
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..utils import generate_tautomer_class_stereobond_aware
    from ..ani import AlchemicalANI1ccx
    import numpy as np

    exp_results = pickle.load(open("data/test_data/exp_results.pickle", "rb"))

    name = "molDWRow_298"
    diameter = 10
    t1_smiles = exp_results[name]["t1-smiles"]
    t2_smiles = exp_results[name]["t2-smiles"]

    # generate both rdkit mol
    t_type, tautomers, flipped = generate_tautomer_class_stereobond_aware(
        name, t1_smiles, t2_smiles, nr_of_conformations=5
    )

    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()
    m = tautomer.add_droplet(
        tautomer.hybrid_topology,
        tautomer.get_hybrid_coordinates(),
        diameter=diameter * unit.angstrom,
        restrain_hydrogen_bonds=True,
        restrain_hydrogen_angles=False,
        top_file=f"data/test_data/{name}_in_droplet.pdb",
    )

    # define the alchemical atoms
    alchemical_atoms = [
        tautomer.hybrid_hydrogen_idx_at_lambda_1,
        tautomer.hybrid_hydrogen_idx_at_lambda_0,
    ]

    # extract hydrogen donor idx and hydrogen idx for from_mol
    model = AlchemicalANI1ccx(alchemical_atoms=alchemical_atoms)
    model = model.to(device)

    # perform initial sampling
    energy_function = neutromeratio.ANI_force_and_energy(
        model=model,
        atoms=tautomer.ligand_in_water_atoms,
        mol=None,
    )

    for r in tautomer.ligand_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)

    for r in tautomer.hybrid_ligand_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)

    x0 = tautomer.get_ligand_in_water_coordinates()
    energy = energy_function.calculate_energy(x0)
    print(energy.energy[0])
    assert is_quantity_close(
        energy.energy[0], (-15146778.81228019 * unit.kilojoule_per_mole)
    )

    tautomer.add_COM_for_hybrid_ligand(
        np.array([diameter / 2, diameter / 2, diameter / 2]) * unit.angstrom
    )

    for r in tautomer.solvent_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)

    for r in tautomer.com_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)

    energy = energy_function.calculate_energy(x0)
    assert is_quantity_close(
        energy.energy[0], (-15018039.455067404 * unit.kilojoule_per_mole)
    )

    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name,
        env="droplet",
        ANImodel=AlchemicalANI1ccx,
        base_path="data/test_data/",
        diameter=diameter,
    )

    energy = energy_function.calculate_energy(x0)
    assert is_quantity_close(
        energy.energy[0], (-15018040.86806798 * unit.kilojoule_per_mole), rtol=1e-7
    )

    del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true",
    reason="Test is failing on travis on MacOS.",
)
def test_thinning():
    from glob import glob

    dcds = glob(f"data/test_data/droplet/molDWRow_298/*dcd")
    top = f"data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    max_snapshots_per_window = 20
    print(dcds)
    for f in dcds:
        print(f)
        traj = md.load_dcd(f, top=top)

        quarter_traj_limit = int(len(traj) / 4)
        snapshots = traj[min(quarter_traj_limit, 10) :].xyz * unit.nanometer
        further_thinning = max(int(len(snapshots) / max_snapshots_per_window), 1)
        snapshots = snapshots[::further_thinning][:max_snapshots_per_window]
        print(len(snapshots))
        assert max_snapshots_per_window == len(snapshots)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true",
    reason="Slow tests are failing on Travis.",
)
def test_max_nr_of_snapshots():
    from ..parameter_gradients import calculate_rmse_between_exp_and_calc
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x

    names = ["molDWRow_298", "SAMPLmol2", "SAMPLmol4"]

    env = "vacuum"

    model = AlchemicalANI1ccx

    for nr_of_snapshots in [20, 80, 120, 150]:
        rmse = calculate_rmse_between_exp_and_calc(
            names,
            model=model,
            data_path=f"./data/test_data/{env}",
            env=env,
            bulk_energy_calculation=True,
            max_snapshots_per_window=nr_of_snapshots,
        )
    del model


def test_unperturbed_perturbed_free_energy():
    # test the setup mbar function with different models, environments and potentials
    from ..parameter_gradients import (
        setup_FEC,
        get_unperturbed_free_energy_difference,
        get_perturbed_free_energy_difference,
    )
    from ..ani import AlchemicalANI2x, CompartimentedAlchemicalANI2x

    name = "molDWRow_298"

    # vacuum
    fec = setup_FEC(
        name,
        env="vacuum",
        data_path="data/test_data/vacuum",
        ANImodel=AlchemicalANI2x,
        bulk_energy_calculation=True,
        max_snapshots_per_window=20,
        load_pickled_FEC=False,
        include_restraint_energy_contribution=True,
        save_pickled_FEC=False,
    )

    a_AlchemicalANI2x = get_unperturbed_free_energy_difference(fec)
    b_AlchemicalANI2x = get_perturbed_free_energy_difference(fec)
    np.isclose(a_AlchemicalANI2x.item(), b_AlchemicalANI2x.item())
    del fec

    # vacuum
    fec = setup_FEC(
        name,
        env="vacuum",
        data_path="data/test_data/vacuum",
        ANImodel=CompartimentedAlchemicalANI2x,
        bulk_energy_calculation=True,
        max_snapshots_per_window=20,
        load_pickled_FEC=False,
        include_restraint_energy_contribution=True,
        save_pickled_FEC=False,
    )

    a_CompartimentedAlchemicalANI2x = get_unperturbed_free_energy_difference(fec)
    b_CompartimentedAlchemicalANI2x = get_perturbed_free_energy_difference(fec)
    np.isclose(
        a_CompartimentedAlchemicalANI2x.item(), b_CompartimentedAlchemicalANI2x.item()
    )
    np.isclose(a_CompartimentedAlchemicalANI2x.item(), b_AlchemicalANI2x.item())
    del fec


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_calculate_rmse_between_exp_and_calc():
    from ..parameter_gradients import (
        calculate_rmse_between_exp_and_calc,
        get_experimental_values,
    )
    from ..constants import kT
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x

    names = ["molDWRow_298", "SAMPLmol2", "SAMPLmol4"]
    exp_results = pickle.load(open("data/test_data/exp_results.pickle", "rb"))

    env = "vacuum"
    exp_values = [get_experimental_values(name) for name in names]

    rmse_list = []
    for model in [AlchemicalANI1ccx, AlchemicalANI2x, AlchemicalANI1x]:
        print(model.name)

        model._reset_parameters()
        rmse, _ = calculate_rmse_between_exp_and_calc(
            names,
            model=model,
            data_path=f"./data/test_data/{env}",
            env=env,
            perturbed_free_energy=True,
            bulk_energy_calculation=True,
            max_snapshots_per_window=100,
        )
        assert np.isclose(
            (exp_results[names[2]]["energy"] * unit.kilocalorie_per_mole) / kT,
            exp_values[2].item(),
        )
        rmse_list.append(rmse)
        model._reset_parameters()
        del model
    print(exp_values)
    print(rmse_list)
    for e1, e2 in zip(
        exp_values,
        [1.8994317488369707, -10.232118388886946, -3.858011851547537],
    ):
        assert np.isclose(e1.item(), e2)

    for e1, e2 in zip(
        rmse_list, [5.662402153015137, 5.6707963943481445, 4.7712321281433105]
    ):
        assert np.isclose(e1, e2)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_calculate_rmse_between_exp_and_calc_droplet():
    from ..parameter_gradients import (
        calculate_rmse_between_exp_and_calc,
        get_experimental_values,
    )
    from ..constants import kT
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x

    rmse_list = []
    names = ["molDWRow_298"]
    exp_results = pickle.load(open("data/test_data/exp_results.pickle", "rb"))
    env = "droplet"
    exp_values = [get_experimental_values(name) for name in names]
    diameter = 10
    assert np.isclose(exp_values[0].item(), 1.8994317488369707)
    assert np.isclose(
        (exp_results[names[0]]["energy"] * unit.kilocalorie_per_mole) / kT,
        exp_values[0].item(),
    )

    for model in [AlchemicalANI1ccx, AlchemicalANI2x, AlchemicalANI1x]:
        model._reset_parameters()
        print(model.name)

        rmse, _ = calculate_rmse_between_exp_and_calc(
            names=names,
            data_path=f"./data/test_data/{env}",
            model=model,
            env=env,
            bulk_energy_calculation=True,
            max_snapshots_per_window=10,
            diameter=diameter,
            perturbed_free_energy=False,
        )
        rmse_list.append(rmse)
        model._reset_parameters()
        del model

    print(rmse_list)
    for e1, e2 in zip(
        rmse_list, [0.23515522480010986, 16.44867706298828, 11.113712310791016]
    ):
        assert np.isclose(e1, e2, rtol=1e-3)


def test_calculate_mse():
    from ..parameter_gradients import calculate_mse
    import torch

    mse = calculate_mse(torch.tensor([1.0]), torch.tensor([4.0]))
    assert mse == 9.0

    mse = calculate_mse(torch.tensor([1.0, 2.0]), torch.tensor([4.0, 2.0]))
    assert mse == 4.5


def test_calculate_rmse():
    from ..parameter_gradients import calculate_rmse
    import numpy as np
    import torch

    rmse = calculate_rmse(torch.tensor([1.0]), torch.tensor([4.0]))
    assert np.isclose(rmse, 3.0)

    rmse = calculate_rmse(torch.tensor([1.0, 2.0]), torch.tensor([4.0, 2.0]))
    assert np.isclose(rmse, 2.1213)

    rmse = calculate_rmse(torch.tensor([0.1, 0.2]), torch.tensor([0.12, 0.24]))
    assert np.isclose(rmse, 0.0316)


def test_bootstrap_tautomer_exp_predict_results():

    from ..analysis import bootstrap_rmse_r

    a1 = np.random.uniform(-1, 0, 10000)
    a2 = np.random.uniform(-1, 0, 10000)
    r = bootstrap_rmse_r(a1, a2, 1000)
    print(r)


def test_experimental_values():
    from ..parameter_gradients import get_experimental_values
    from ..constants import _get_names

    def compare_get_names():
        from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds

        with open("data/test_data/exp_results.pickle", "rb") as f:
            exp_results = pickle.load(f)
        names = []
        for name in sorted(exp_results):
            if name in exclude_set_ANI + mols_with_charge + multiple_stereobonds:
                continue
            names.append(name)
        return names

    assert _get_names() == compare_get_names()
    names = _get_names()
    n_list = torch.stack([get_experimental_values(name) for name in names])
    assert len(n_list) == len(names)
