"""
Unit and regression test for the neutromeratio package.
"""

# Import package, test suite, and other packages as needed
import neutromeratio
import pytest
import pickle
import torch
from simtk import unit
from neutromeratio.constants import device
from neutromeratio.utils import _get_traj
import torchani
from openmmtools.utils import is_quantity_close
from neutromeratio.constants import device


def test_tochani_neutromeratio_sync():

    """
    Make sure that ANI in neutromeratio and in torchani return the same energy
    """
    import torch
    import torchani
    from ..ani import ANI1ccx, ANI1x, ANI2x

    device = torch.device("cpu")
    model_torchani_ANI2x = torchani.models.ANI2x(periodic_table_index=True).to(device)
    model_torchani_ANI1ccx = torchani.models.ANI1ccx(periodic_table_index=True).to(
        device
    )
    model_torchani_ANIx = torchani.models.ANI1x(periodic_table_index=True).to(device)

    model_neutromeratio_ANI1ccx = ANI1ccx(periodic_table_index=True).to(device)
    model_neutromeratio_ANI1x = ANI1x(periodic_table_index=True).to(device)
    model_neutromeratio_ANI2x = ANI2x(periodic_table_index=True).to(device)

    coordinates = torch.tensor(
        [
            [
                [0.03192167, 0.00638559, 0.01301679],
                [-0.83140486, 0.39370209, -0.26395324],
                [-0.66518241, -0.84461308, 0.20759389],
                [0.45554739, 0.54289633, 0.81170881],
                [0.66091919, -0.16799635, -0.91037834],
            ]
        ],
        requires_grad=True,
        device=device,
    )
    # In periodic table, C = 6 and H = 1
    species = torch.tensor([[6, 1, 1, 1, 1]], device=device)

    for model_torchani, model_neutromeratio in zip(
        [model_torchani_ANIx, model_torchani_ANI1ccx, model_torchani_ANI2x],
        [
            model_neutromeratio_ANI1x,
            model_neutromeratio_ANI1ccx,
            model_neutromeratio_ANI2x,
        ],
    ):
        energy_torchani = model_torchani((species, coordinates)).energies
        e1 = energy_torchani.item()
        print("Energy:", energy_torchani.item())

        energy_neutromeratio = model_neutromeratio(
            (species, coordinates, True)
        ).energies
        print("Energy:", energy_neutromeratio.item())
        e2 = energy_neutromeratio.item()
        assert e1 == e2


def test_neutromeratio_energy_calculations_with_ANI_in_vacuum():

    from ..tautomers import Tautomer
    from ..constants import kT
    from ..analysis import setup_alchemical_system_and_energy_function
    import numpy as np
    from ..ani import AlchemicalANI1ccx, ANI1ccx, ANI2x

    # read in exp_results.pickle
    with open("data/test_data/exp_results.pickle", "rb") as f:
        exp_results = pickle.load(f)

    # vacuum system
    name = "molDWRow_298"
    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name, ANImodel=AlchemicalANI1ccx, env="vacuum"
    )

    # read in pregenerated traj
    traj_path = (
        "data/test_data/vacuum/molDWRow_298/molDWRow_298_lambda_0.0000_in_vacuum.dcd"
    )
    top_path = "data/test_data/vacuum/molDWRow_298/molDWRow_298.pdb"
    traj, _ = _get_traj(traj_path, top_path, [tautomer.hybrid_hydrogen_idx_at_lambda_0])
    coordinates = [x.xyz[0] for x in traj[:10]] * unit.nanometer

    # test energies with neutromeratio ANI objects
    # first with ANI1ccx
    model = neutromeratio.ani.ANI1ccx()
    torch.set_num_threads(1)
    energy_function = neutromeratio.ANI_force_and_energy(
        model=model, atoms=tautomer.initial_state_ligand_atoms, mol=None
    )
    energy = energy_function.calculate_energy(coordinates)

    assert is_quantity_close(
        energy.energy[0].in_units_of(unit.kilojoule_per_mole),
        (-906555.29945346 * unit.kilojoule_per_mole),
        rtol=1e-5,
    )

    # with ANI2x
    model = neutromeratio.ani.ANI2x()
    torch.set_num_threads(1)
    energy_function = neutromeratio.ANI_force_and_energy(
        model=model, atoms=tautomer.initial_state_ligand_atoms, mol=None
    )
    energy = energy_function.calculate_energy(coordinates)

    assert is_quantity_close(
        energy.energy[0].in_units_of(unit.kilojoule_per_mole),
        (-907243.8987177598 * unit.kilojoule_per_mole),
        rtol=1e-5,
    )


def test_neutromeratio_energy_calculations_with_AlchemicalANI1ccx():
    from ..tautomers import Tautomer
    import numpy as np
    from ..constants import kT
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x, ANI1ccx

    # read in exp_results.pickle
    with open("data/test_data/exp_results.pickle", "rb") as f:
        exp_results = pickle.load(f)

    ######################################################################
    # vacuum
    ######################################################################
    name = "molDWRow_298"
    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name, env="vacuum", ANImodel=AlchemicalANI1ccx
    )
    # read in pregenerated traj
    traj_path = (
        "data/test_data/vacuum/molDWRow_298/molDWRow_298_lambda_0.0000_in_vacuum.dcd"
    )
    top_path = "data/test_data/vacuum/molDWRow_298/molDWRow_298.pdb"
    traj, _ = _get_traj(traj_path, top_path)

    x0 = [x.xyz[0] for x in traj[:10]] * unit.nanometer

    energy_1 = energy_function.calculate_energy(x0, lambda_value=1.0)
    for e1, e2 in zip(
        energy_1.energy,
        [
            -906555.29945346,
            -905750.20471091,
            -906317.24952004,
            -906545.17543265,
            -906581.65215098,
            -906618.2832786,
            -906565.05631782,
            -905981.82167316,
            -904681.20632002,
            -904296.8214631,
        ]
        * unit.kilojoule_per_mole,
    ):
        assert is_quantity_close(e1, e2, rtol=1e-2)


def test_neutromeratio_energy_calculations_with_ANI_in_droplet():
    from ..tautomers import Tautomer
    from ..constants import kT
    from ..analysis import setup_alchemical_system_and_energy_function
    import numpy as np
    from ..ani import AlchemicalANI1ccx

    # read in exp_results.pickle
    with open("data/test_data/exp_results.pickle", "rb") as f:
        exp_results = pickle.load(f)

    # droplet system
    name = "molDWRow_298"
    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name,
        ANImodel=AlchemicalANI1ccx,
        env="droplet",
        diameter=10,
        base_path="data/test_data/droplet/molDWRow_298/",
    )

    # read in pregenerated traj
    traj_path = (
        "data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
    )
    top_path = "data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    # _get_traj remove the dummy atom
    traj, _ = _get_traj(traj_path, top_path, [tautomer.hybrid_hydrogen_idx_at_lambda_0])
    # overwrite the coordinates that rdkit generated with the first frame in the traj
    torch.set_num_threads(1)
    coordinates = [x.xyz[0] for x in traj[:10]] * unit.nanometer

    # removes the dummy atom
    atoms = (
        tautomer.ligand_in_water_atoms[: tautomer.hybrid_hydrogen_idx_at_lambda_0]
        + tautomer.ligand_in_water_atoms[tautomer.hybrid_hydrogen_idx_at_lambda_0 + 1 :]
    )
    assert len(tautomer.ligand_in_water_atoms) == len(atoms) + 1

    # ANI1ccx
    model = neutromeratio.ani.ANI1ccx()
    energy_function = neutromeratio.ANI_force_and_energy(
        model=model, atoms=atoms, mol=None
    )
    energy = energy_function.calculate_energy(coordinates)

    assert is_quantity_close(
        energy.energy[0].in_units_of(unit.kilojoule_per_mole),
        (-3514015.2561722626 * unit.kilojoule_per_mole),
        rtol=1e-5,
    )

    # ANI2x
    model = neutromeratio.ani.ANI2x()
    atoms = (
        tautomer.ligand_in_water_atoms[: tautomer.hybrid_hydrogen_idx_at_lambda_0]
        + tautomer.ligand_in_water_atoms[tautomer.hybrid_hydrogen_idx_at_lambda_0 + 1 :]
    )
    assert len(tautomer.ligand_in_water_atoms) == len(atoms) + 1
    energy_function = neutromeratio.ANI_force_and_energy(
        model=model, atoms=atoms, mol=None
    )
    energy = energy_function.calculate_energy(coordinates)

    assert is_quantity_close(
        energy.energy[0].in_units_of(unit.kilojoule_per_mole),
        (-3515114.528875586 * unit.kilojoule_per_mole),
        rtol=1e-5,
    )
