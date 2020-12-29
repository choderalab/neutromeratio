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


def test_setup_energy_function():
    # test the seupup of the energy function with different alchemical potentials
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI2x, ANI1ccx

    name = "molDWRow_298"

    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name, env="vacuum", ANImodel=AlchemicalANI2x
    )
    assert flipped == True

    failed = False
    try:
        # this should to fail
        (
            energy_function,
            tautomer,
            flipped,
        ) = setup_alchemical_system_and_energy_function(
            name=name, env="vacuum", ANImodel=ANI1ccx
        )
    except RuntimeError:
        failed = True
        pass

    # make sure that setup_alchemical_system_and_energy_function has failed with non-alchemical potential
    assert failed == True


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_memory_issue():
    # test the seup of the energy function with different alchemical potentials
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI2x, ANI1ccx
    from glob import glob
    from ..constants import device
    import torchani, torch

    # either use neutromeratio ANI object
    nn = ANI1ccx(periodic_table_index=True).to(device)
    # or torchani ANI object
    nn = torchani.models.ANI1ccx(periodic_table_index=True).to(device)

    # read in trajs
    name = "molDWRow_298"
    data_path = "data/test_data/droplet/"
    max_snapshots_per_window = 5
    dcds = glob(f"{data_path}/{name}/*.dcd")

    md_trajs = []
    energies = []
    e = 0.0
    # read in all the frames from the trajectories
    top = f"{data_path}/{name}/{name}_in_droplet.pdb"

    species = []
    for dcd_filename in dcds:
        traj = md.load_dcd(dcd_filename, top=top)
        snapshots = traj.xyz * unit.nanometer
        further_thinning = max(int(len(snapshots) / max_snapshots_per_window), 1)
        snapshots = snapshots[::further_thinning][:max_snapshots_per_window]
        coordinates = [sample / unit.angstrom for sample in snapshots] * unit.angstrom
        md_trajs.append(coordinates)

    # generate species
    for a in traj.topology.atoms:
        species.append(a.element.symbol)

    element_index = {"C": 6, "N": 7, "O": 8, "H": 1}
    species = [element_index[e] for e in species]

    # calcualte energies
    for traj in md_trajs:
        print(len(traj))
        for xyz in traj:
            coordinates = torch.tensor(
                [xyz.value_in_unit(unit.nanometer)],
                requires_grad=False,
                device=device,
                dtype=torch.float32,
            )

            species_tensor = torch.tensor(
                [species] * len(coordinates), device=device, requires_grad=False
            )
            energy = nn((species_tensor, coordinates)).energies
            energies.append(energy.item())
            e += energy.item()
    np.isclose(e, 69309.46893170934, rtol=1e-9)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_setup_FEC():
    # test the setup mbar function with different models, environments and potentials
    from ..parameter_gradients import setup_FEC
    from ..ani import (
        AlchemicalANI2x,
        AlchemicalANI1x,
        AlchemicalANI1ccx,
        CompartimentedAlchemicalANI2x,
    )

    name = "molDWRow_298"

    # vacuum
    fec = setup_FEC(
        name,
        env="vacuum",
        data_path="data/test_data/vacuum",
        ANImodel=AlchemicalANI1ccx,
        bulk_energy_calculation=False,
        max_snapshots_per_window=20,
        load_pickled_FEC=False,
        include_restraint_energy_contribution=True,
        save_pickled_FEC=False,
    )
    assert np.isclose(-3.2194223855155357, fec._end_state_free_energy_difference[0])

    # AlchemicalANI2x
    fec = setup_FEC(
        name,
        env="vacuum",
        data_path="data/test_data/vacuum",
        ANImodel=AlchemicalANI2x,
        bulk_energy_calculation=False,
        max_snapshots_per_window=20,
        load_pickled_FEC=False,
        include_restraint_energy_contribution=True,
        save_pickled_FEC=False,
    )
    assert np.isclose(-11.554636171428106, fec._end_state_free_energy_difference[0])

    # CompartimentedAlchemicalANI2x
    fec = setup_FEC(
        name,
        env="vacuum",
        data_path="data/test_data/vacuum",
        ANImodel=CompartimentedAlchemicalANI2x,
        bulk_energy_calculation=False,
        max_snapshots_per_window=20,
        load_pickled_FEC=False,
        include_restraint_energy_contribution=True,
        save_pickled_FEC=False,
    )
    assert np.isclose(-11.554636171428106, fec._end_state_free_energy_difference[0])

    fec = setup_FEC(
        name,
        env="vacuum",
        data_path="data/test_data/vacuum",
        ANImodel=AlchemicalANI1x,
        bulk_energy_calculation=False,
        max_snapshots_per_window=20,
        load_pickled_FEC=False,
        include_restraint_energy_contribution=True,
        save_pickled_FEC=False,
    )
    assert np.isclose(-12.413598945128637, fec._end_state_free_energy_difference[0])

    # droplet
    fec = setup_FEC(
        name,
        env="droplet",
        diameter=10,
        data_path="data/test_data/droplet",
        ANImodel=AlchemicalANI1ccx,
        bulk_energy_calculation=False,
        max_snapshots_per_window=10,
        load_pickled_FEC=False,
        include_restraint_energy_contribution=True,
        save_pickled_FEC=False,
    )
    assert np.isclose(-1.6642793589801324, fec._end_state_free_energy_difference[0])

    # AlchemicalANI2x
    fec = setup_FEC(
        name,
        env="droplet",
        diameter=10,
        data_path="data/test_data/droplet",
        ANImodel=AlchemicalANI2x,
        bulk_energy_calculation=False,
        max_snapshots_per_window=10,
        load_pickled_FEC=False,
        include_restraint_energy_contribution=True,
        save_pickled_FEC=False,
    )
    assert np.isclose(-18.348107633661936, fec._end_state_free_energy_difference[0])

    # CompartimentedAlchemicalANI2x
    fec = setup_FEC(
        name,
        env="droplet",
        diameter=10,
        data_path="data/test_data/droplet",
        ANImodel=CompartimentedAlchemicalANI2x,
        bulk_energy_calculation=False,
        max_snapshots_per_window=10,
        load_pickled_FEC=False,
        include_restraint_energy_contribution=True,
        save_pickled_FEC=False,
    )
    assert np.isclose(-18.348107633661936, fec._end_state_free_energy_difference[0])

    fec = setup_FEC(
        name,
        env="droplet",
        diameter=10,
        data_path="data/test_data/droplet",
        ANImodel=AlchemicalANI1x,
        bulk_energy_calculation=False,
        max_snapshots_per_window=10,
        load_pickled_FEC=False,
        include_restraint_energy_contribution=True,
        save_pickled_FEC=False,
    )
    assert np.isclose(-13.013142148962665, fec._end_state_free_energy_difference[0])

    for testdir in [
        f"data/test_data/droplet/{name}",
        f"data/test_data/vacuum/{name}",
    ]:
        # remove the pickle files
        for item in os.listdir(testdir):
            if item.endswith(".pickle"):
                os.remove(os.path.join(testdir, item))


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_setup_FEC_test_pickle_files():
    # test the setup mbar function, write out the pickle file and test that everything works
    from ..parameter_gradients import setup_FEC
    from ..ani import AlchemicalANI2x, AlchemicalANI1x, AlchemicalANI1ccx

    test = os.listdir("data/test_data/vacuum")

    name = "molDWRow_298"
    model = AlchemicalANI1ccx
    # remove the pickle files
    for testdir in [
        f"data/test_data/droplet/{name}",
        f"data/test_data/vacuum/{name}",
    ]:
        # remove the pickle files
        for item in os.listdir(testdir):
            if item.endswith(".pickle"):
                os.remove(os.path.join(testdir, item))

    # vacuum
    fec = setup_FEC(
        name,
        env="vacuum",
        data_path="data/test_data/vacuum",
        ANImodel=model,
        bulk_energy_calculation=False,
        max_snapshots_per_window=20,
        load_pickled_FEC=False,
        include_restraint_energy_contribution=True,
    )
    assert np.isclose(-3.2194223855155357, fec._end_state_free_energy_difference[0])

    # vacuum
    fec = setup_FEC(
        name,
        env="vacuum",
        data_path="data/test_data/vacuum",
        ANImodel=model,
        bulk_energy_calculation=False,
        max_snapshots_per_window=20,
        load_pickled_FEC=True,
        include_restraint_energy_contribution=True,
    )
    model._reset_parameters()
    del model
    assert np.isclose(-3.2194223855155357, fec._end_state_free_energy_difference[0])


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow calculation."
)
def test_FEC_with_different_free_energy_calls():
    from ..parameter_gradients import (
        get_perturbed_free_energy_difference,
        get_unperturbed_free_energy_difference,
    )
    from ..constants import kT, device, exclude_set_ANI, mols_with_charge
    from ..parameter_gradients import setup_FEC
    from glob import glob
    from ..ani import (
        AlchemicalANI1ccx,
        AlchemicalANI1x,
        AlchemicalANI2x,
        CompartimentedAlchemicalANI2x,
    )
    import numpy as np

    env = "vacuum"
    names = ["molDWRow_298", "SAMPLmol2"]

    ################################
    model = AlchemicalANI1ccx

    # testing fec calculation
    bulk_energy_calculation = True
    fec_list = [
        setup_FEC(
            name,
            ANImodel=model,
            env=env,
            bulk_energy_calculation=bulk_energy_calculation,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=80,
            load_pickled_FEC=False,
            include_restraint_energy_contribution=True,
            save_pickled_FEC=False,
        )
        for name in names
    ]

    # no modifications to the potential
    # all the following calls should return the same value
    assert len(fec_list) == 2

    # look at the two functions from parameter_gradient
    # both of these correct for the flipped energy calculation
    # and flip the sign of the prediction of mol298
    fec_values = get_perturbed_free_energy_difference(fec_list)
    print(fec_values)
    for e1, e2 in zip(fec_values, [1.2104, -5.3161]):
        assert np.isclose(e1.item(), e2, rtol=1e-2)

    fec_values = get_unperturbed_free_energy_difference(fec_list)
    print(fec_values)
    for e1, e2 in zip(fec_values, [1.2104, -5.3161]):
        assert np.isclose(e1.item(), e2, rtol=1e-2)

    fec_values = [
        fec_list[0]._end_state_free_energy_difference[0],
        fec_list[1]._end_state_free_energy_difference[0],
    ]
    print(fec_values)
    for e1, e2 in zip(fec_values, [-1.2104, -5.3161]):
        assert np.isclose(e1.item(), e2, rtol=1e-2)

    fec_values = (
        fec_list[0]._compute_free_energy_difference(),
        fec_list[1]._compute_free_energy_difference(),
    )
    print(fec_values)
    for e1, e2 in zip(fec_values, [-1.2104, -5.3161]):
        assert np.isclose(e1.item(), e2, rtol=1e-2)

    ################################
    model = AlchemicalANI2x

    # testing fec calculation
    bulk_energy_calculation = True
    fec_list = [
        setup_FEC(
            name,
            ANImodel=model,
            env=env,
            bulk_energy_calculation=bulk_energy_calculation,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=80,
            load_pickled_FEC=False,
            include_restraint_energy_contribution=True,
            save_pickled_FEC=False,
        )
        for name in names
    ]

    # no modifications to the potential
    # all the following calls should return the same value
    assert len(fec_list) == 2

    # look at the two functions from parameter_gradient
    # both of these correct for the flipped energy calculation
    # and flip the sign of the prediction of mol298
    fec_values = get_perturbed_free_energy_difference(fec_list)
    print(fec_values)
    for e1, e2 in zip(fec_values, [8.7152, -9.2873]):
        assert np.isclose(e1.item(), e2, rtol=1e-2)

    fec_values = get_unperturbed_free_energy_difference(fec_list)
    print(fec_values)
    for e1, e2 in zip(fec_values, [8.7152, -9.2873]):
        assert np.isclose(e1.item(), e2, rtol=1e-2)

    fec_values = [
        fec_list[0]._end_state_free_energy_difference[0],
        fec_list[1]._end_state_free_energy_difference[0],
    ]
    print(fec_values)
    for e1, e2 in zip(fec_values, [-8.7152, -9.2873]):
        assert np.isclose(e1.item(), e2, rtol=1e-2)

    fec_values = (
        fec_list[0]._compute_free_energy_difference(),
        fec_list[1]._compute_free_energy_difference(),
    )
    print(fec_values)
    for e1, e2 in zip(fec_values, [-8.7152, -9.2873]):
        assert np.isclose(e1.item(), e2, rtol=1e-2)

    ################################
    model = CompartimentedAlchemicalANI2x

    # testing fec calculation
    bulk_energy_calculation = True
    fec_list = [
        setup_FEC(
            name,
            ANImodel=model,
            env=env,
            bulk_energy_calculation=bulk_energy_calculation,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=80,
            load_pickled_FEC=False,
            include_restraint_energy_contribution=True,
            save_pickled_FEC=False,
        )
        for name in names
    ]

    # no modifications to the potential
    # all the following calls should return the same value
    assert len(fec_list) == 2

    # look at the two functions from parameter_gradient
    # both of these correct for the flipped energy calculation
    # and flip the sign of the prediction of mol298
    fec_values = get_perturbed_free_energy_difference(fec_list)
    print(fec_values)
    for e1, e2 in zip(fec_values, [8.7152, -9.2873]):
        assert np.isclose(e1.item(), e2, rtol=1e-2)

    fec_values = get_unperturbed_free_energy_difference(fec_list)
    print(fec_values)
    for e1, e2 in zip(fec_values, [8.7152, -9.2873]):
        assert np.isclose(e1.item(), e2, rtol=1e-2)

    fec_values = [
        fec_list[0]._end_state_free_energy_difference[0],
        fec_list[1]._end_state_free_energy_difference[0],
    ]
    print(fec_values)
    for e1, e2 in zip(fec_values, [-8.7152, -9.2873]):
        assert np.isclose(e1.item(), e2, rtol=1e-2)

    fec_values = (
        fec_list[0]._compute_free_energy_difference(),
        fec_list[1]._compute_free_energy_difference(),
    )
    print(fec_values)
    for e1, e2 in zip(fec_values, [-8.7152, -9.2873]):
        assert np.isclose(e1.item(), e2, rtol=1e-2)
    del fec_list
    del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow calculation."
)
def test_FEC_with_perturbed_free_energies():
    from ..parameter_gradients import get_perturbed_free_energy_difference
    from ..constants import kT, device
    from ..parameter_gradients import setup_FEC
    from ..ani import (
        AlchemicalANI1ccx,
        AlchemicalANI1x,
        AlchemicalANI2x,
        CompartimentedAlchemicalANI2x,
    )
    import numpy as np

    env = "vacuum"
    names = ["molDWRow_298", "SAMPLmol2"]

    for idx, model in enumerate(
        [
            AlchemicalANI1ccx,
            AlchemicalANI1x,
            AlchemicalANI2x,
            CompartimentedAlchemicalANI2x,
        ]
    ):
        model._reset_parameters()

        # AlchemicalANI1ccx
        if idx == 0:
            # testing fec calculation in sequence
            bulk_energy_calculation = True
            fec_list = [
                setup_FEC(
                    name,
                    ANImodel=model,
                    env=env,
                    bulk_energy_calculation=bulk_energy_calculation,
                    data_path="./data/test_data/vacuum",
                    max_snapshots_per_window=80,
                    load_pickled_FEC=False,
                    include_restraint_energy_contribution=True,
                    save_pickled_FEC=False,
                )
                for name in names
            ]

            assert len(fec_list) == 2
            fec_values = get_perturbed_free_energy_difference(fec_list)
            for e1, e2 in zip(fec_values, [1.2104, -5.3161]):
                assert np.isclose(e1.item(), e2, rtol=1e-4)

            # testing fec calculation in bulk
            bulk_energy_calculation = False
            fec_list = [
                setup_FEC(
                    name,
                    ANImodel=model,
                    env=env,
                    bulk_energy_calculation=bulk_energy_calculation,
                    data_path="./data/test_data/vacuum",
                    max_snapshots_per_window=80,
                    load_pickled_FEC=False,
                    include_restraint_energy_contribution=True,
                    save_pickled_FEC=False,
                )
                for name in names
            ]

            assert len(fec_list) == 2
            fec = get_perturbed_free_energy_difference(fec_list)
            print(fec)
            for e1, e2 in zip(fec, [1.2104, -5.3161]):
                assert np.isclose(e1.item(), e2, rtol=1e-4)

            fec = setup_FEC(
                "molDWRow_298",
                ANImodel=model,
                env=env,
                bulk_energy_calculation=True,
                data_path="./data/test_data/vacuum",
                max_snapshots_per_window=80,
                load_pickled_FEC=False,
                include_restraint_energy_contribution=True,
                save_pickled_FEC=False,
            )
            assert np.isclose(
                fec._end_state_free_energy_difference[0],
                fec._compute_free_energy_difference().item(),
                rtol=1e-5,
            )

        # AlchemicalANI1x
        if idx == 1:
            bulk_energy_calculation = True
            fec_list = [
                setup_FEC(
                    name,
                    ANImodel=model,
                    env=env,
                    bulk_energy_calculation=bulk_energy_calculation,
                    data_path="./data/test_data/vacuum",
                    max_snapshots_per_window=60,
                    load_pickled_FEC=False,
                    include_restraint_energy_contribution=True,
                    save_pickled_FEC=False,
                )
                for name in names
            ]

            assert len(fec_list) == 2
            fec = get_perturbed_free_energy_difference(fec_list)
            print(fec)
            for e1, e2 in zip(fec, [10.3192, -9.746403840249418]):
                assert np.isclose(e1.item(), e2, rtol=1e-4)

        # AlchemicalANI2x
        if idx == 2:
            bulk_energy_calculation = True
            fec_list = [
                setup_FEC(
                    name,
                    ANImodel=model,
                    env=env,
                    bulk_energy_calculation=bulk_energy_calculation,
                    data_path="./data/test_data/vacuum",
                    max_snapshots_per_window=60,
                    load_pickled_FEC=False,
                    include_restraint_energy_contribution=True,
                    save_pickled_FEC=False,
                )
                for name in names
            ]

            assert len(fec_list) == 2
            fec = get_perturbed_free_energy_difference(fec_list)
            print(fec)
            for e1, e2 in zip(fec, [8.8213, -9.664895714083166]):
                assert np.isclose(e1.item(), e2, rtol=1e-4)

        # CompartimentedAlchemicalANI2x
        if idx == 3:
            bulk_energy_calculation = True
            fec_list = [
                setup_FEC(
                    name,
                    ANImodel=model,
                    env=env,
                    bulk_energy_calculation=bulk_energy_calculation,
                    data_path="./data/test_data/vacuum",
                    max_snapshots_per_window=60,
                    load_pickled_FEC=False,
                    include_restraint_energy_contribution=True,
                    save_pickled_FEC=False,
                )
                for name in names
            ]

            assert len(fec_list) == 2
            fec = get_perturbed_free_energy_difference(fec_list)
            print(fec)
            for e1, e2 in zip(fec, [8.8213, -9.664895714083166]):
                assert np.isclose(e1.item(), e2, rtol=1e-4)
        del fec_list
        del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow calculation."
)
def test_FEC_with_perturbed_free_energies_with_and_without_restraints():
    from ..parameter_gradients import get_perturbed_free_energy_difference
    from ..constants import kT, device, exclude_set_ANI, mols_with_charge
    from ..parameter_gradients import setup_FEC
    from glob import glob
    from ..ani import AlchemicalANI2x
    import numpy as np

    env = "vacuum"
    names = ["molDWRow_298", "SAMPLmol2"]

    model = AlchemicalANI2x

    bulk_energy_calculation = True
    fec_list = [
        setup_FEC(
            name,
            ANImodel=model,
            env=env,
            bulk_energy_calculation=bulk_energy_calculation,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=60,
            load_pickled_FEC=False,
            include_restraint_energy_contribution=True,
            save_pickled_FEC=False,
        )
        for name in names
    ]

    assert len(fec_list) == 2
    fec = get_perturbed_free_energy_difference(fec_list)
    print(fec)
    for e1, e2 in zip(fec, [8.8213, -9.6649]):
        assert np.isclose(e1.item(), e2, rtol=1e-4)

    fec_list = [
        setup_FEC(
            name,
            ANImodel=model,
            env=env,
            bulk_energy_calculation=bulk_energy_calculation,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=60,
            load_pickled_FEC=False,
            include_restraint_energy_contribution=False,
            save_pickled_FEC=False,
        )
        for name in names
    ]

    assert len(fec_list) == 2
    fec = get_perturbed_free_energy_difference(fec_list)
    print(fec)
    for e1, e2 in zip(fec, [8.8213, -9.6649]):
        assert np.isclose(e1.item(), e2, rtol=1e-4)
    del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_loading_saving_mbar_object_AlchemicalANI2x():
    # test the setup mbar function with different models, environments and potentials
    from ..parameter_gradients import (
        setup_FEC,
        _load_checkpoint,
        _get_nn_layers,
        get_perturbed_free_energy_difference,
    )
    from ..ani import AlchemicalANI2x

    AlchemicalANI2x._reset_parameters()
    model_name = "AlchemicalANI2x"
    name = "molDWRow_298"
    model_instance = AlchemicalANI2x([0, 0])
    AdamW, AdamW_scheduler, SGD, SGD_scheduler = _get_nn_layers(
        nr_of_nn=8, ANImodel=model_instance, elements="CHON"
    )
    # initial parameters
    params1 = list(model_instance.optimized_neural_network.parameters())[6][0].tolist()

    # setup FEC and save pickle file
    fec = setup_FEC(
        name,
        env="droplet",
        diameter=10,
        data_path="data/test_data/droplet",
        ANImodel=AlchemicalANI2x,
        bulk_energy_calculation=True,
        max_snapshots_per_window=10,
        load_pickled_FEC=False,
        include_restraint_energy_contribution=True,
        save_pickled_FEC=True,
    )
    assert np.isclose(
        18.348107633661936, get_perturbed_free_energy_difference([fec]).tolist()[0]
    )

    # load checkpoint parameter file and override optimized parameters
    _load_checkpoint(
        f"data/test_data/{model_name}_3.pt",
        model_instance,
        AdamW,
        AdamW_scheduler,
        SGD,
        SGD_scheduler,
    )
    # make sure that initial parameters are new parameters
    params2 = list(model_instance.optimized_neural_network.parameters())[6][0].tolist()
    assert params1 != params2

    # get new free energy
    assert np.isclose(
        3.2730393726044866, get_perturbed_free_energy_difference([fec]).tolist()[0]
    )
    del fec

    # load FEC object --> the FEC should now use the optimized parameters!
    fec = setup_FEC(
        name,
        env="droplet",
        diameter=10,
        data_path="data/test_data/droplet",
        ANImodel=AlchemicalANI2x,
        bulk_energy_calculation=True,
        max_snapshots_per_window=10,
        load_pickled_FEC=True,
        include_restraint_energy_contribution=True,
        save_pickled_FEC=False,
    )

    assert np.isclose(
        3.2730393726044866, get_perturbed_free_energy_difference([fec]).tolist()[0]
    )

    pickled_model = fec.ani_model.model
    params3 = list(pickled_model.optimized_neural_network.parameters())[6][0].tolist()

    assert params2 == params3
    del fec
    del model_instance


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_loading_saving_mbar_object_CompartimentedAlchemicalANI2x():
    # test the setup mbar function with different models, environments and potentials
    from ..parameter_gradients import (
        setup_FEC,
        _load_checkpoint,
        _get_nn_layers,
        get_perturbed_free_energy_difference,
    )
    from ..ani import CompartimentedAlchemicalANI2x

    CompartimentedAlchemicalANI2x._reset_parameters()
    name = "molDWRow_298"
    model_instance = CompartimentedAlchemicalANI2x([0, 0])
    AdamW, AdamW_scheduler, SGD, SGD_scheduler = _get_nn_layers(
        nr_of_nn=8, ANImodel=model_instance, elements="CHON"
    )
    # initial parameters
    params1 = list(model_instance.optimized_neural_network.parameters())[6][0].tolist()

    # droplet
    fec = setup_FEC(
        name,
        env="droplet",
        diameter=10,
        data_path="data/test_data/droplet",
        ANImodel=CompartimentedAlchemicalANI2x,
        bulk_energy_calculation=True,
        max_snapshots_per_window=10,
        load_pickled_FEC=False,
        include_restraint_energy_contribution=True,
        save_pickled_FEC=True,
    )
    assert np.isclose(
        18.348107633661936, get_perturbed_free_energy_difference(fec).tolist()[0]
    )

    # load checkpoint parameter file and override optimized parameters
    _load_checkpoint(
        f"data/test_data/AlchemicalANI2x_3.pt",
        model_instance,
        AdamW,
        AdamW_scheduler,
        SGD,
        SGD_scheduler,
    )
    # make sure that initial parameters are new parameters
    params2 = list(model_instance.optimized_neural_network.parameters())[6][0].tolist()
    assert params1 != params2
    # get new free energy
    assert np.isclose(
        3.2730393726044866,
        get_perturbed_free_energy_difference(fec).tolist()[0],  # -1.3759686627878307
    )
    del fec

    # load FEC object --> the FEC should now use the optimized parameters!
    fec = setup_FEC(
        name,
        env="droplet",
        diameter=10,
        data_path="data/test_data/droplet",
        ANImodel=CompartimentedAlchemicalANI2x,
        bulk_energy_calculation=True,
        max_snapshots_per_window=10,
        load_pickled_FEC=True,
        include_restraint_energy_contribution=True,
        save_pickled_FEC=False,
    )

    assert np.isclose(
        3.2730393726044866, get_perturbed_free_energy_difference(fec).tolist()[0]
    )

    pickled_model = fec.ani_model.model
    params3 = list(pickled_model.optimized_neural_network.parameters())[6][0].tolist()

    assert params2 == params3
    del fec
    del model_instance


def test_io_checkpoints():
    from ..parameter_gradients import _load_checkpoint, _get_nn_layers
    from ..ani import (
        AlchemicalANI1ccx,
        AlchemicalANI1x,
        AlchemicalANI2x,
        CompartimentedAlchemicalANI2x,
    )

    # specify the system you want to simulate
    for idx, (model, model_name) in enumerate(
        zip(
            [
                AlchemicalANI1ccx,
                AlchemicalANI2x,
                AlchemicalANI1x,
                CompartimentedAlchemicalANI2x,
            ],
            [
                "AlchemicalANI1ccx",
                "AlchemicalANI2x",
                "AlchemicalANI1x",
                "CompartimentedAlchemicalANI2x",
            ],
        )
    ):
        model._reset_parameters()
        # set tweaked parameters
        print(model_name)
        model_instance = model([0, 0])
        AdamW, AdamW_scheduler, SGD, SGD_scheduler = _get_nn_layers(
            8, model_instance, elements="CHON"
        )
        # initial parameters
        params1 = list(model.optimized_neural_network.parameters())[6][0].tolist()

        # load parameters
        _load_checkpoint(
            f"data/test_data/{model_name}_3.pt",
            model_instance,
            AdamW,
            AdamW_scheduler,
            SGD,
            SGD_scheduler,
        )
        params2 = list(model_instance.optimized_neural_network.parameters())[6][
            0
        ].tolist()
        # make sure somehting happend
        assert params1 != params2
        # test that new instances have the new parameters
        m = model([0, 0])
        params3 = list(m.optimized_neural_network.parameters())[6][0].tolist()
        assert params2 == params3
        model._reset_parameters()
        del model


def test_load_parameters():
    from ..parameter_gradients import _save_checkpoint, _load_checkpoint, _get_nn_layers
    from ..ani import (
        AlchemicalANI1ccx,
        AlchemicalANI1x,
        AlchemicalANI2x,
        CompartimentedAlchemicalANI2x,
    )

    # specify the system you want to simulate
    for idx, (model, model_name) in enumerate(
        zip(
            [
                AlchemicalANI1ccx,
                AlchemicalANI2x,
                AlchemicalANI1x,
                CompartimentedAlchemicalANI2x,
            ],
            [
                "AlchemicalANI1ccx",
                "AlchemicalANI2x",
                "AlchemicalANI1x",
                "CompartimentedAlchemicalANI2x",
            ],
        )
    ):
        model._reset_parameters()
        # set tweaked parameters
        model_instance = model([0, 0])
        # initial parameters
        params1 = list(model_instance.original_neural_network.parameters())[6][
            0
        ].tolist()
        params2 = list(model_instance.optimized_neural_network.parameters())[6][
            0
        ].tolist()
        assert params1 == params2
        # load parameters
        model_instance.load_nn_parameters(f"data/test_data/{model_name}_3.pt")
        params1 = list(model_instance.original_neural_network.parameters())[6][
            0
        ].tolist()
        params2 = list(model_instance.optimized_neural_network.parameters())[6][
            0
        ].tolist()
        # make sure somehting happend
        assert params1 != params2
        # test that new instances have the new parameters
        m = model([0, 0])
        params3 = list(m.optimized_neural_network.parameters())[6][0].tolist()
        assert params2 == params3
        model._reset_parameters()
        del model
