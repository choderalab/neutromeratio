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
import torchani
from openmmtools.utils import is_quantity_close
import pandas as pd
from rdkit import Chem
import pytest_benchmark
from neutromeratio.constants import device


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
@pytest.mark.benchmark(min_rounds=1, warmup=False)
def test_timing_for_perturebed_free_energy_u_ln(benchmark):
    from ..parameter_gradients import get_perturbed_free_energy_difference, setup_FEC
    from ..ani import AlchemicalANI2x
    import os

    model = AlchemicalANI2x
    max_snapshots_per_window = 100
    names = ["molDWRow_298"]
    env = "droplet"
    diameter = 10
    model._reset_parameters()
    m = model([0, 0])
    torch.set_num_threads(4)

    name = "molDWRow_298"
    # precalcualte mbar
    fec = setup_FEC(
        name,
        env="droplet",
        diameter=10,
        data_path="data/test_data/droplet",
        ANImodel=model,
        bulk_energy_calculation=True,  # doesn't matter, since we are loading the reuslts from disk
        max_snapshots_per_window=max_snapshots_per_window,
        load_pickled_FEC=True,
    )

    def wrap():
        fec._form_u_ln()

    benchmark.pedantic(wrap, rounds=1, iterations=3)
    del fec


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
@pytest.mark.benchmark(min_rounds=1, warmup=False)
def test_timing_for_perturebed_free_energy_u_ln_and_perturbed_free_energy(benchmark):
    from ..parameter_gradients import get_perturbed_free_energy_difference, setup_FEC
    from ..ani import AlchemicalANI2x
    import os

    model = AlchemicalANI2x
    model._reset_parameters()

    max_snapshots_per_window = 100
    names = ["molDWRow_298"]
    env = "droplet"
    diameter = 10
    m = model([0, 0])
    torch.set_num_threads(4)

    name = "molDWRow_298"
    # precalcualte mbar
    fec = setup_FEC(
        name,
        env="droplet",
        diameter=10,
        data_path="data/test_data/droplet",
        ANImodel=model,
        bulk_energy_calculation=True,  # doesn't matter, since we are loading the reuslts from disk
        max_snapshots_per_window=max_snapshots_per_window,
        load_pickled_FEC=True,
    )

    def wrap():
        get_perturbed_free_energy_difference([fec])

    benchmark.pedantic(wrap, rounds=1, iterations=3)
    del fec


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
@pytest.mark.benchmark(min_rounds=3)
def test_timing_for_single_energy_calculation_with_AlchemicalANI_10_snapshots_batch(
    benchmark,
):
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI2x

    # NOTE: Sometimes this test fails? something wrong with molDWRow_68?
    from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds
    import random, shutil
    from ..constants import _get_names
    from simtk import unit

    torch.set_num_threads(1)

    names = _get_names()
    lambda_value = 0.1
    name = "molDWRow_298"

    (energy_function, tautomer, flipped,) = setup_alchemical_system_and_energy_function(
        name=name,
        env="droplet",
        ANImodel=AlchemicalANI2x,
        base_path=f"data/test_data/droplet/{name}",
        diameter=10,
    )
    traj_path = (
        "data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
    )
    top_path = "data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    traj, top = _get_traj(traj_path, top_path, None)
    coordinates = [x.xyz[0] for x in traj[:10]] * unit.nanometer

    def wrap1():
        e = energy_function.calculate_energy(coordinates, lambda_value)
        return e

    e = benchmark(wrap1)
    for e_pre, e_cal in zip(
        [
            -3515574.05857072,
            -3515478.60995353,
            -3515367.0878032,
            -3515332.90224507,
            -3515360.70976201,
            -3515465.75272167,
            -3515465.71963145,
            -3515456.76306932,
            -3515458.36516877,
            -3515457.31727224,
        ],
        e.energy.value_in_unit(unit.kilojoule_per_mole),
    ):
        np.isclose(e_pre, e_cal, rtol=1e-5)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
@pytest.mark.benchmark(min_rounds=2)
def test_timing_for_single_energy_calculation_with_AlchemicalANI_1_snapshot_batch(
    benchmark,
):
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x

    # NOTE: Sometimes this test fails? something wrong with molDWRow_68?
    from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds
    import random, shutil
    from ..constants import _get_names

    names = _get_names()
    lambda_value = 0.1
    name = "molDWRow_298"

    (energy_function, tautomer, flipped,) = setup_alchemical_system_and_energy_function(
        name=name,
        env="droplet",
        ANImodel=AlchemicalANI2x,
        base_path=f"data/test_data/droplet/{name}",
        diameter=10,
    )
    traj_path = (
        "data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
    )
    top_path = "data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    traj, top = _get_traj(traj_path, top_path, None)
    coordinates = [x.xyz[0] for x in traj[:10]] * unit.nanometer

    def wrap1():
        energy_function.calculate_energy(coordinates, lambda_value)

    coordinates = [x.xyz[0] for x in traj[1]] * unit.nanometer

    benchmark(wrap1)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
@pytest.mark.benchmark(min_rounds=2)
def test_timing_for_single_energy_calculation_with_AlchemicalANI_20_snapshot_batch(
    benchmark,
):
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x

    # NOTE: Sometimes this test fails? something wrong with molDWRow_68?
    from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds
    import random, shutil
    from ..constants import _get_names

    names = _get_names()
    lambda_value = 0.1
    name = "molDWRow_298"

    (energy_function, tautomer, flipped,) = setup_alchemical_system_and_energy_function(
        name=name,
        env="droplet",
        ANImodel=AlchemicalANI2x,
        base_path=f"data/test_data/droplet/{name}",
        diameter=10,
    )
    traj_path = (
        "data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
    )
    top_path = "data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    traj, top = _get_traj(traj_path, top_path, None)
    coordinates = [x.xyz[0] for x in traj[:20]] * unit.nanometer

    def wrap1():
        energy_function.calculate_energy(coordinates, lambda_value)

    benchmark(wrap1)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
@pytest.mark.benchmark(min_rounds=4)
def test_timing_for_single_energy_calculation_with_AlchemicalANI_100_snapshot_batch(
    benchmark,
):
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x

    # NOTE: Sometimes this test fails? something wrong with molDWRow_68?
    from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds
    import random, shutil
    from ..constants import _get_names

    names = _get_names()
    lambda_value = 0.1
    name = "molDWRow_298"
    torch.set_num_threads(1)

    (energy_function, tautomer, flipped,) = setup_alchemical_system_and_energy_function(
        name=name,
        env="droplet",
        ANImodel=AlchemicalANI2x,
        base_path=f"data/test_data/droplet/{name}",
        diameter=10,
    )
    traj_path = (
        "data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
    )
    top_path = "data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    traj, top = _get_traj(traj_path, top_path, None)
    coordinates = [x.xyz[0] for x in traj[:100]] * unit.nanometer

    def wrap1():
        energy_function.calculate_energy(coordinates, lambda_value)

    benchmark(wrap1)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
@pytest.mark.benchmark(min_rounds=2)
def test_timing_for_single_energy_calculation_with_AlchemicalANI_200_snapshot_batch(
    benchmark,
):
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x

    # NOTE: Sometimes this test fails? something wrong with molDWRow_68?
    from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds
    import random, shutil
    from ..constants import _get_names

    torch.set_num_threads(1)

    names = _get_names()
    lambda_value = 0.1
    name = "molDWRow_298"

    (energy_function, tautomer, flipped,) = setup_alchemical_system_and_energy_function(
        name=name,
        env="droplet",
        ANImodel=AlchemicalANI2x,
        base_path=f"data/test_data/droplet/{name}",
        diameter=10,
    )
    traj_path = (
        "data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
    )
    top_path = "data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    traj, top = _get_traj(traj_path, top_path, None)
    coordinates = [x.xyz[0] for x in traj[:200]] * unit.nanometer

    def wrap1():
        energy_function.calculate_energy(coordinates, lambda_value)

    benchmark(wrap1)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
@pytest.mark.benchmark(min_rounds=2)
def test_timing_for_single_energy_calculation_with_AlchemicalANI_500_snapshot_batch(
    benchmark,
):
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x

    # NOTE: Sometimes this test fails? something wrong with molDWRow_68?
    from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds
    import random, shutil
    from ..constants import _get_names

    torch.set_num_threads(1)

    names = _get_names()
    lambda_value = 0.1
    name = "molDWRow_298"

    (energy_function, tautomer, flipped,) = setup_alchemical_system_and_energy_function(
        name=name,
        env="droplet",
        ANImodel=AlchemicalANI2x,
        base_path=f"data/test_data/droplet/{name}",
        diameter=10,
    )
    traj_path = (
        "data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
    )
    top_path = "data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    traj, top = _get_traj(traj_path, top_path, None)
    coordinates = [x.xyz[0] for x in traj[:500]] * unit.nanometer

    def wrap1():
        energy_function.calculate_energy(coordinates, lambda_value)

    benchmark(wrap1)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
@pytest.mark.benchmark(min_rounds=2)
def test_timing_for_single_energy_calculation_with_AlchemicalANI_1100_snapshot_batch(
    benchmark,
):
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x

    # NOTE: Sometimes this test fails? something wrong with molDWRow_68?
    from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds
    import random, shutil
    from ..constants import _get_names

    torch.set_num_threads(1)

    names = _get_names()
    lambda_value = 0.1
    name = "molDWRow_298"

    (energy_function, tautomer, flipped,) = setup_alchemical_system_and_energy_function(
        name=name,
        env="droplet",
        ANImodel=AlchemicalANI2x,
        base_path=f"data/test_data/droplet/{name}",
        diameter=10,
    )
    traj_path = (
        "data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
    )
    top_path = "data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    traj, top = _get_traj(traj_path, top_path, None)
    coordinates = [x.xyz[0] for x in traj[:1100]] * unit.nanometer

    def wrap1():
        energy_function.calculate_energy(coordinates, lambda_value)

    benchmark(wrap1)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
@pytest.mark.benchmark(min_rounds=3)
def test_timing_for_single_energy_calculation_with_AlchemicalANI_1100_snapshot_batch_lambda_0(
    benchmark,
):
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x

    # NOTE: Sometimes this test fails? something wrong with molDWRow_68?
    from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds
    import random, shutil
    from ..constants import _get_names

    torch.set_num_threads(4)

    names = _get_names()
    lambda_value = 0.0
    name = "molDWRow_298"

    (energy_function, tautomer, flipped,) = setup_alchemical_system_and_energy_function(
        name=name,
        env="droplet",
        ANImodel=AlchemicalANI2x,
        base_path=f"data/test_data/droplet/{name}",
        diameter=10,
    )
    traj_path = (
        "data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
    )
    top_path = "data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    traj, top = _get_traj(traj_path, top_path, None)
    coordinates = [x.xyz[0] for x in traj[:1100]] * unit.nanometer

    def wrap1():
        energy_function.calculate_energy(coordinates, lambda_value)

    benchmark(wrap1)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
@pytest.mark.benchmark(min_rounds=3)
def test_timing_for_single_energy_calculation_with_AlchemicalANI_1100_snapshot_batch_lambda_1(
    benchmark,
):
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x

    # NOTE: Sometimes this test fails? something wrong with molDWRow_68?
    from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds
    import random, shutil
    from ..constants import _get_names

    torch.set_num_threads(4)

    names = _get_names()
    lambda_value = 1.0
    name = "molDWRow_298"

    (energy_function, tautomer, flipped,) = setup_alchemical_system_and_energy_function(
        name=name,
        env="droplet",
        ANImodel=AlchemicalANI2x,
        base_path=f"data/test_data/droplet/{name}",
        diameter=10,
    )
    traj_path = (
        "data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
    )
    top_path = "data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    traj, top = _get_traj(traj_path, top_path, None)
    coordinates = [x.xyz[0] for x in traj[:1100]] * unit.nanometer

    def wrap1():
        energy_function.calculate_energy(coordinates, lambda_value)

    benchmark(wrap1)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
@pytest.mark.benchmark(min_rounds=2)
def test_timing_for_single_energy_calculation_with_AlchemicalANI_20_snapshot_sequence(
    benchmark,
):
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x

    # NOTE: Sometimes this test fails? something wrong with molDWRow_68?
    from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds
    import random, shutil
    from ..constants import _get_names

    names = _get_names()
    lambda_value = 0.1
    name = "molDWRow_298"

    (energy_function, tautomer, flipped,) = setup_alchemical_system_and_energy_function(
        name=name,
        env="droplet",
        ANImodel=AlchemicalANI2x,
        base_path=f"data/test_data/droplet/{name}",
        diameter=10,
    )
    traj_path = (
        "data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
    )
    top_path = "data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    traj, top = _get_traj(traj_path, top_path, None)
    coordinates = [x.xyz[0] for x in traj[:20]] * unit.nanometer

    def wrap1():
        for c in coordinates:
            c = c.value_in_unit(unit.nanometer)
            c = [c] * unit.nanometer
            energy_function.calculate_energy(c, lambda_value)

    benchmark(wrap1)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
@pytest.mark.benchmark(min_rounds=2)
def test_timing_for_single_energy_calculation_with_AlchemicalANI_100_snapshot_sequence(
    benchmark,
):
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x

    # NOTE: Sometimes this test fails? something wrong with molDWRow_68?
    from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds
    import random, shutil
    from ..constants import _get_names

    torch.set_num_threads(4)

    names = _get_names()
    lambda_value = 0.1
    name = "molDWRow_298"

    (energy_function, tautomer, flipped,) = setup_alchemical_system_and_energy_function(
        name=name,
        env="droplet",
        ANImodel=AlchemicalANI2x,
        base_path=f"data/test_data/droplet/{name}",
        diameter=10,
    )
    traj_path = (
        "data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
    )
    top_path = "data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    traj, top = _get_traj(traj_path, top_path, None)
    coordinates = [x.xyz[0] for x in traj[:100]] * unit.nanometer

    def wrap1():
        for c in coordinates:
            c = c.value_in_unit(unit.nanometer)
            c = [c] * unit.nanometer
            energy_function.calculate_energy(c, lambda_value)

    benchmark(wrap1)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
@pytest.mark.benchmark(min_rounds=2)
def test_timing_for_single_energy_calculation_with_AlchemicalANI_200_snapshot_sequence(
    benchmark,
):
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x

    # NOTE: Sometimes this test fails? something wrong with molDWRow_68?
    from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds
    import random, shutil
    from ..constants import _get_names

    torch.set_num_threads(4)

    names = _get_names()
    lambda_value = 0.1
    name = "molDWRow_298"

    (energy_function, tautomer, flipped,) = setup_alchemical_system_and_energy_function(
        name=name,
        env="droplet",
        ANImodel=AlchemicalANI2x,
        base_path=f"data/test_data/droplet/{name}",
        diameter=10,
    )
    traj_path = (
        "data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
    )
    top_path = "data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    traj, top = _get_traj(traj_path, top_path, None)
    coordinates = [x.xyz[0] for x in traj[:200]] * unit.nanometer

    def wrap1():
        for c in coordinates:
            c = c.value_in_unit(unit.nanometer)
            c = [c] * unit.nanometer
            energy_function.calculate_energy(c, lambda_value)

    benchmark(wrap1)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
@pytest.mark.benchmark(min_rounds=1, warmup=False)
def test_timing_main_training_loop_without_pickled_tautomer_object(benchmark):
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining,
    )
    from ..ani import AlchemicalANI2x
    import os

    model = AlchemicalANI2x
    max_snapshots_per_window = 50
    names = ["molDWRow_298"]
    env = "droplet"
    diameter = 10
    max_epochs = 5
    torch.set_num_threads(1)
    name = "molDWRow_298"
    # remove the pickle files
    for testdir in [
        f"data/test_data/droplet/{name}",
    ]:
        # remove the pickle files
        for item in os.listdir(testdir):
            if item.endswith(".pickle"):
                print(item)
                os.remove(os.path.join(testdir, item))

    def wrap():

        rmse_val = setup_and_perform_parameter_retraining(
            env=env,
            names_training=names,
            names_validating=names,
            ANImodel=model,
            batch_size=1,
            max_snapshots_per_window=max_snapshots_per_window,
            data_path=f"./data/test_data/{env}",
            nr_of_nn=8,
            max_epochs=max_epochs,
            diameter=diameter,
            checkpoint_filename=f"AlchemicalANI2x_droplet.pt",
            load_checkpoint=False,
            bulk_energy_calculation=False,
            load_pickled_FEC=False,
        )
        print(rmse_val)

    benchmark.pedantic(wrap, rounds=1, iterations=1)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
@pytest.mark.benchmark(min_rounds=1, warmup=False)
def test_timing_main_training_loop_with_pickled_tautomer_object(benchmark):
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining,
    )
    from ..ani import AlchemicalANI2x
    import os

    model = AlchemicalANI2x
    max_snapshots_per_window = 100
    names = ["molDWRow_298"]
    env = "droplet"
    diameter = 10
    max_epochs = 5
    torch.set_num_threads(1)
    name = "molDWRow_298"
    # remove the pickle files
    for testdir in [
        f"data/test_data/droplet/{name}",
    ]:
        # remove the pickle files
        for item in os.listdir(testdir):
            if item.endswith(".pickle"):
                print(item)
                os.remove(os.path.join(testdir, item))

    def wrap():

        rmse_val = setup_and_perform_parameter_retraining(
            env=env,
            names_training=names,
            names_validating=names,
            ANImodel=model,
            batch_size=1,
            max_snapshots_per_window=max_snapshots_per_window,
            data_path=f"./data/test_data/{env}",
            nr_of_nn=8,
            max_epochs=max_epochs,
            diameter=diameter,
            checkpoint_filename=f"AlchemicalANI2x_droplet.pt",
            load_checkpoint=False,
            bulk_energy_calculation=False,
            load_pickled_FEC=True,
        )
        print(rmse_val)

    benchmark.pedantic(wrap, rounds=1, iterations=1)
