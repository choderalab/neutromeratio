"""
Unit and regression test for the neutromeratio package.
"""

# Import package, test suite, and other packages as needed
from typing import Tuple
import neutromeratio
import pytest
import os
import torch
from simtk import unit
import numpy as np
import mdtraj as md
from neutromeratio.constants import device
from openmmtools.utils import is_quantity_close
from neutromeratio.constants import device


def _get_params(model, layer: int) -> Tuple[list, list]:
    layer = -1
    weight_layers = []
    bias_layers = []

    for nn in model:
        weight_layers.extend(
            [
                nn.C[layer].weight.tolist(),
                nn.H[layer].weight.tolist(),
                nn.O[layer].weight.tolist(),
                nn.N[layer].weight.tolist(),
            ]
        )
        bias_layers.extend(
            [
                nn.C[layer].bias.tolist(),
                nn.H[layer].bias.tolist(),
                nn.O[layer].bias.tolist(),
                nn.N[layer].bias.tolist(),
            ]
        )
    return weight_layers, bias_layers


def _remove_files(name, max_epochs=1):
    try:
        os.remove(f"{name}.pt")
    except FileNotFoundError:
        pass
    for i in range(1, max_epochs):
        os.remove(f"{name}_{i}.pt")
    os.remove(f"{name}_best.pt")


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_postprocessing_droplet():
    from ..parameter_gradients import (
        get_perturbed_free_energy_difference,
        get_experimental_values,
    )
    from ..constants import kT, device, exclude_set_ANI, mols_with_charge
    from ..parameter_gradients import setup_FEC
    from glob import glob
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x

    for idx, model in enumerate([AlchemicalANI1ccx, AlchemicalANI1x]):
        print(model.name)
        model._reset_parameters()

        env = "droplet"
        names = ["molDWRow_298"]
        diameter = 10

        # include restraints, don't load from pickle
        fec_list = [
            setup_FEC(
                name,
                ANImodel=model,
                env=env,
                diameter=diameter,
                bulk_energy_calculation=True,
                data_path="data/test_data/droplet",
                max_snapshots_per_window=10,
                load_pickled_FEC=False,
                include_restraint_energy_contribution=True,
                save_pickled_FEC=True,
            )
            for name in names
        ]

        if idx == 0:
            print(fec_list)
            assert len(fec_list) == 1
            rmse = torch.sqrt(
                torch.mean(
                    (
                        get_perturbed_free_energy_difference(fec_list)
                        - get_experimental_values(names)
                    )
                    ** 2
                )
            )
            assert np.isclose(
                fec_list[0]._end_state_free_energy_difference[0].item(),
                -1.664279359190441,
                rtol=1e-3,
            )
            assert np.isclose(rmse.item(), 0.23514418557176664, rtol=1e-3)

        elif idx == 1:
            print(fec_list)
            assert len(fec_list) == 1
            rmse = torch.sqrt(
                torch.mean(
                    (
                        get_perturbed_free_energy_difference(fec_list)
                        - get_experimental_values(names)
                    )
                    ** 2
                )
            )
            assert np.isclose(
                fec_list[0]._end_state_free_energy_difference[0].item(),
                -13.013144575760862,
                rtol=1e-3,
            )
            assert np.isclose(
                rmse.item(),
                11.11371282694961,
                rtol=1e-3,
            )
        # exclude restraints, don't load from pickle
        fec_list = [
            setup_FEC(
                name,
                ANImodel=model,
                env=env,
                diameter=diameter,
                bulk_energy_calculation=True,
                data_path="data/test_data/droplet",
                max_snapshots_per_window=10,
                include_restraint_energy_contribution=False,
                load_pickled_FEC=False,
                save_pickled_FEC=True,
            )
            for name in names
        ]

        if idx == 0:
            print(fec_list)
            assert len(fec_list) == 1
            rmse = torch.sqrt(
                torch.mean(
                    (
                        get_perturbed_free_energy_difference(fec_list)
                        - get_experimental_values(names)
                    )
                    ** 2
                )
            )
            assert np.isclose(
                fec_list[0]._end_state_free_energy_difference[0].item(),
                -1.664279359190441,
                rtol=1e-3,
            )
            assert np.isclose(rmse.item(), 0.23514418557176664, rtol=1e-3)

        elif idx == 1:
            print(fec_list)
            assert len(fec_list) == 1
            rmse = torch.sqrt(
                torch.mean(
                    (
                        get_perturbed_free_energy_difference(fec_list)
                        - get_experimental_values(names)
                    )
                    ** 2
                )
            )
            assert np.isclose(
                fec_list[0]._end_state_free_energy_difference[0].item(),
                -13.013144575760862,
                rtol=1e-3,
            )
            assert np.isclose(
                rmse.item(),
                11.11371282694961,
                rtol=1e-3,
            )

        # incldue restraints, load from pickle
        fec_list = [
            setup_FEC(
                name,
                ANImodel=model,
                env=env,
                diameter=diameter,
                bulk_energy_calculation=True,
                data_path="data/test_data/droplet",
                max_snapshots_per_window=10,
                include_restraint_energy_contribution=True,
                load_pickled_FEC=True,
                save_pickled_FEC=True,
            )
            for name in names
        ]

        if idx == 0:
            print(fec_list)
            assert len(fec_list) == 1
            rmse = torch.sqrt(
                torch.mean(
                    (
                        get_perturbed_free_energy_difference(fec_list)
                        - get_experimental_values(names)
                    )
                    ** 2
                )
            )
            assert np.isclose(
                fec_list[0]._end_state_free_energy_difference[0].item(),
                -1.664279359190441,
                rtol=1e-3,
            )
            assert np.isclose(rmse.item(), 0.23514418557176664, rtol=1e-3)

        elif idx == 1:
            print(fec_list)
            assert len(fec_list) == 1
            rmse = torch.sqrt(
                torch.mean(
                    (
                        get_perturbed_free_energy_difference(fec_list)
                        - get_experimental_values(names)
                    )
                    ** 2
                )
            )
            assert np.isclose(
                fec_list[0]._end_state_free_energy_difference[0].item(),
                -13.013144575760862,
                rtol=1e-3,
            )
            assert np.isclose(
                rmse.item(),
                11.11371282694961,
                rtol=1e-3,
            )

        # exclude restraints, load from pickle
        fec_list = [
            setup_FEC(
                name,
                ANImodel=model,
                env=env,
                diameter=diameter,
                bulk_energy_calculation=True,
                data_path="data/test_data/droplet",
                max_snapshots_per_window=10,
                include_restraint_energy_contribution=False,
                load_pickled_FEC=True,
            )
            for name in names
        ]

        if idx == 0:
            print(fec_list)
            assert len(fec_list) == 1
            rmse = torch.sqrt(
                torch.mean(
                    (
                        get_perturbed_free_energy_difference(fec_list)
                        - get_experimental_values(names)
                    )
                    ** 2
                )
            )
            assert np.isclose(
                fec_list[0]._end_state_free_energy_difference[0].item(),
                -1.664279359190441,
                rtol=1e-3,
            )
            assert np.isclose(rmse.item(), 0.23514418557176664, rtol=1e-3)

        elif idx == 1:
            print(fec_list)
            assert len(fec_list) == 1
            rmse = torch.sqrt(
                torch.mean(
                    (
                        get_perturbed_free_energy_difference(fec_list)
                        - get_experimental_values(names)
                    )
                    ** 2
                )
            )
            assert np.isclose(
                fec_list[0]._end_state_free_energy_difference[0].item(),
                -13.013144575760862,
                rtol=1e-3,
            )
            assert np.isclose(
                rmse.item(),
                11.11371282694961,
                rtol=1e-3,
            )

    del fec_list


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_tweak_parameters_and_class_nn_AlchemicalANI():
    # the tweaked parameters are stored as class variables
    # this can lead to some tricky situations.
    # It also means that whenever any optimization is performed,
    # every new instance of the class has the new parameters
    from ..parameter_gradients import setup_and_perform_parameter_retraining
    from ..ani import AlchemicalANI1ccx, CompartimentedAlchemicalANI2x

    names = ["molDWRow_298"]
    max_epochs = 3
    layer = -1

    # tweak parameters
    for model, model_name in zip(
        (AlchemicalANI1ccx, CompartimentedAlchemicalANI2x),
        ("AlchemicalANI1ccx", "CompartimentedAlchemicalANI2x"),
    ):
        model._reset_parameters()
        layer = -1
        print(model)
        # start with model
        model_instance = model([0, 0])
        # save parameters at the beginning
        optimized_neural_network_before_retraining = _get_params(
            model_instance.optimized_neural_network, layer
        )
        original_neural_network_before_retraining = _get_params(
            model_instance.original_neural_network, layer
        )

        # optimized equals original parameters
        assert (
            optimized_neural_network_before_retraining
            == original_neural_network_before_retraining
        )

        _ = setup_and_perform_parameter_retraining(
            env="vacuum",
            checkpoint_filename=f"{model_name}_vacuum.pt",
            names_training=names,
            names_validating=names,
            ANImodel=model,
            max_snapshots_per_window=50,
            batch_size=1,
            data_path="./data/test_data/vacuum",
            nr_of_nn=8,
            load_checkpoint=False,
            max_epochs=max_epochs,
            load_pickled_FEC=False,
        )

        _remove_files(f"{model_name}_vacuum", max_epochs)
        # get new optimized parameters
        optimized_neural_network_after_retraining = _get_params(
            model_instance.optimized_neural_network, layer
        )
        # make sure that somethign happend while tweaking
        assert (
            optimized_neural_network_before_retraining
            != optimized_neural_network_after_retraining
        )

        # new initialize second model
        new_model_instance = model([0, 0])
        # get original parameters
        new_original_neural_network_before_retraining = _get_params(
            new_model_instance.original_neural_network, layer
        )
        # get tweaked parameters
        new_optimized_neural_network_before_retraining = _get_params(
            new_model_instance.optimized_neural_network, layer
        )
        # optimized parameters at start of model2 should be the same as at end of model1
        assert (
            new_optimized_neural_network_before_retraining
            == optimized_neural_network_after_retraining
        )
        # optimzied parameters at start of model 1 are different than at start of model2
        assert (
            new_optimized_neural_network_before_retraining
            != optimized_neural_network_before_retraining
        )
        # original parameters are the same
        assert (
            new_original_neural_network_before_retraining
            == original_neural_network_before_retraining
        )
        model._reset_parameters()


def test_tweak_parameters_and_class_nn_CompartimentedAlchemicalANI2x():
    # the tweaked parameters are stored as class variables
    # this can lead to some tricky situations.
    # It also means that whenever any optimization is performed,
    # every new instance of the class has the new parameters
    from ..parameter_gradients import setup_and_perform_parameter_retraining
    from ..ani import CompartimentedAlchemicalANI2x

    model, model_name = (CompartimentedAlchemicalANI2x, "CompartimentedAlchemicalANI2x")
    model._reset_parameters()

    names = ["molDWRow_298"]
    max_epochs = 3
    layer = -1
    model_instance = model([0, 0])
    # save parameters at the beginning

    optimized_neural_network_parameters_before_training = _get_params(
        model_instance.optimized_neural_network, layer
    )
    original_neural_network_parameters_before_training = _get_params(
        model_instance.original_neural_network, layer
    )
    assert (
        optimized_neural_network_parameters_before_training
        == original_neural_network_parameters_before_training
    )
    _ = setup_and_perform_parameter_retraining(
        env="vacuum",
        checkpoint_filename=f"{model_name}_vacuum.pt",
        names_training=names,
        names_validating=names,
        ANImodel=model,
        max_snapshots_per_window=10,
        batch_size=1,
        data_path="./data/test_data/vacuum",
        nr_of_nn=8,
        load_checkpoint=False,
        max_epochs=max_epochs,
        load_pickled_FEC=True,
    )

    _remove_files(f"{model_name}_vacuum", max_epochs)
    # get optimized parameters
    optimized_neural_network_after_retraining = _get_params(
        model_instance.optimized_neural_network, layer
    )
    original_neural_network_after_retraining = _get_params(
        model_instance.original_neural_network, layer
    )

    # make sure that somethign happend while tweaking
    assert (
        optimized_neural_network_parameters_before_training
        != optimized_neural_network_after_retraining
    )

    assert (
        original_neural_network_parameters_before_training
        == original_neural_network_after_retraining
    )

    new_model_instance = model([0, 0])

    new_optimized_neural_network_parameters_before_training = _get_params(
        new_model_instance.optimized_neural_network, layer
    )
    new_original_neural_network_parameters_before_training = _get_params(
        new_model_instance.original_neural_network, layer
    )

    # original parameters stay the same
    assert (
        new_original_neural_network_parameters_before_training
        == original_neural_network_parameters_before_training
    )
    assert (
        original_neural_network_after_retraining
        == new_original_neural_network_parameters_before_training
    )
    # new optimized parameters before retraining are the same as opt after retraining
    assert (
        new_optimized_neural_network_parameters_before_training
        == optimized_neural_network_after_retraining
    )


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_tweak_parameters_vacuum_multiple_tautomer():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
    )
    import os
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x

    # calculate with batch_size=3
    names = ["molDWRow_298", "SAMPLmol2", "SAMPLmol4"]
    max_epochs = 4
    for idx, (model, model_name) in enumerate(
        zip(
            [AlchemicalANI1ccx, AlchemicalANI2x, AlchemicalANI1x],
            ["AlchemicalANI1ccx", "AlchemicalANI2x", "AlchemicalANI1x"],
        )
    ):

        model._reset_parameters()
        (
            rmse_val,
            rmse_test,
        ) = setup_and_perform_parameter_retraining_with_test_set_split(
            env="vacuum",
            checkpoint_filename=f"{model_name}_vacuum.pt",
            names=names,
            ANImodel=model,
            batch_size=3,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=50,
            nr_of_nn=8,
            bulk_energy_calculation=True,
            max_epochs=max_epochs,
            load_checkpoint=False,
            load_pickled_FEC=False,
        )

        if idx == 0:
            print(rmse_val)
            try:
                assert np.isclose(rmse_val[-1], rmse_test)
                assert np.isclose(rmse_val[0], 5.3938140869140625)
                assert np.isclose(rmse_val[-1], 2.098975658416748)
            finally:
                _remove_files(model_name + "_vacuum", max_epochs)

        if idx == 1:
            print(rmse_val)
            try:

                assert np.isclose(rmse_val[-1], rmse_test)
                assert np.isclose(rmse_val[0], 5.187891006469727)
                assert np.isclose(rmse_val[-1], 2.672308921813965)
            finally:
                _remove_files(model_name + "_vacuum", max_epochs)

        if idx == 2:
            print(rmse_val)
            try:
                assert np.isclose(rmse_val[-1], rmse_test)
                assert np.isclose(rmse_val[0], 4.582426071166992)
                assert np.isclose(rmse_val[-1], 2.2336010932922363)
            finally:
                _remove_files(model_name + "_vacuum", max_epochs)
        model._reset_parameters()


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_tweak_parameters_vacuum_single_tautomer_AlchemicalANI2x():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
    )
    from ..ani import AlchemicalANI2x

    # calculate with batch_size=1
    # load pickled tautomer object
    names = ["molDWRow_298"]
    max_epochs = 4
    for model, model_name in zip(
        [AlchemicalANI2x],
        ["AlchemicalANI2x"],
    ):
        model._reset_parameters()

        (
            rmse_val,
            rmse_test,
        ) = setup_and_perform_parameter_retraining_with_test_set_split(
            env="vacuum",
            checkpoint_filename=f"{model_name}_vacuum.pt",
            names=names,
            ANImodel=model,
            batch_size=1,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=50,
            nr_of_nn=8,
            bulk_energy_calculation=True,
            max_epochs=max_epochs,
            load_checkpoint=False,
        )

        print(rmse_val)
        try:
            assert np.isclose(rmse_val[-1], rmse_test)
            assert np.isclose(rmse_val[0], 5.7811503410339355)
            assert np.isclose(rmse_val[-1], 2.1603381633758545)
        finally:
            _remove_files(model_name + "_vacuum", max_epochs)

    # without pickled tautomer object
    names = ["molDWRow_298"]
    max_epochs = 4
    for model, model_name in zip(
        [AlchemicalANI2x],
        ["AlchemicalANI2x"],
    ):
        model._reset_parameters()

        (
            rmse_val,
            rmse_test,
        ) = setup_and_perform_parameter_retraining_with_test_set_split(
            env="vacuum",
            checkpoint_filename=f"{model_name}_vacuum.pt",
            names=names,
            ANImodel=model,
            batch_size=1,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=50,
            nr_of_nn=8,
            bulk_energy_calculation=True,
            max_epochs=max_epochs,
            load_checkpoint=False,
            load_pickled_FEC=False,
        )

        print(rmse_val)
        try:
            assert np.isclose(rmse_val[-1], rmse_test)
            assert np.isclose(rmse_val[0], 5.7811503410339355)
            assert np.isclose(rmse_val[-1], 2.1603381633758545)
        finally:
            _remove_files(model_name + "_vacuum", max_epochs)

        model._reset_parameters()
        del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_tweak_parameters_vacuum_single_tautomer_CompartimentedAlchemicalANI2x():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
    )
    from ..ani import CompartimentedAlchemicalANI2x

    # without pickled tautomer object
    names = ["molDWRow_298"]
    max_epochs = 4
    for model, model_name in zip(
        [CompartimentedAlchemicalANI2x],
        ["CompartimentedAlchemicalANI2x"],
    ):
        model._reset_parameters()

        (
            rmse_val,
            rmse_test,
        ) = setup_and_perform_parameter_retraining_with_test_set_split(
            env="vacuum",
            checkpoint_filename=f"{model_name}_vacuum.pt",
            names=names,
            ANImodel=model,
            batch_size=1,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=50,
            nr_of_nn=8,
            bulk_energy_calculation=True,
            max_epochs=max_epochs,
            load_checkpoint=False,
            load_pickled_FEC=False,
        )

        print(rmse_val)
        try:
            assert np.isclose(rmse_val[-1], rmse_test)
            assert np.isclose(rmse_val[0], 5.7811503410339355)
            assert np.isclose(rmse_val[-1], 2.1603381633758545)
        finally:
            pass
            # _remove_files(model_name + "_vacuum", max_epochs)

        model._reset_parameters()
        del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_tweak_parameters_vacuum_single_tautomer_CompartimentedAlchemicalANI2x_load_FEC():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
    )
    from ..ani import CompartimentedAlchemicalANI2x

    # without pickled tautomer object
    names = ["molDWRow_298"]
    max_epochs = 4
    for model, model_name in zip(
        [CompartimentedAlchemicalANI2x],
        ["CompartimentedAlchemicalANI2x"],
    ):
        model._reset_parameters()

        (
            rmse_val,
            rmse_test,
        ) = setup_and_perform_parameter_retraining_with_test_set_split(
            env="vacuum",
            checkpoint_filename=f"{model_name}_vacuum.pt",
            names=names,
            ANImodel=model,
            batch_size=1,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=50,
            nr_of_nn=8,
            bulk_energy_calculation=True,
            max_epochs=max_epochs,
            load_checkpoint=False,
            load_pickled_FEC=True,
        )

        print(rmse_val)
        try:
            assert np.isclose(rmse_val[-1], rmse_test)
            assert np.isclose(rmse_val[0], 5.7811503410339355)
            assert np.isclose(rmse_val[-1], 2.1603381633758545)
        finally:
            _remove_files(model_name + "_vacuum", max_epochs)

        del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_tweak_parameters_droplet_with_AlchemicalANI():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
    )
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x

    names = ["molDWRow_298"]
    env = "droplet"
    diameter = 10
    max_epochs = 3
    for idx, (model, model_name) in enumerate(
        zip(
            [AlchemicalANI1ccx, AlchemicalANI2x, AlchemicalANI1x],
            ["AlchemicalANI1ccx", "AlchemicalANI2x", "AlchemicalANI1x"],
        )
    ):

        model._reset_parameters()

        (
            rmse_val,
            rmse_test,
        ) = setup_and_perform_parameter_retraining_with_test_set_split(
            env=env,
            names=names,
            ANImodel=model,
            batch_size=1,
            max_snapshots_per_window=10,
            checkpoint_filename=f"{model_name}_droplet.pt",
            data_path=f"./data/test_data/{env}",
            nr_of_nn=8,
            max_epochs=max_epochs,
            diameter=diameter,
            load_checkpoint=False,
            load_pickled_FEC=False,
        )

        if idx == 0:
            try:
                assert np.isclose(rmse_val[-1], rmse_test)
                assert np.isclose(rmse_val[0], 0.23515522480010986)
                assert np.isclose(rmse_val[-1], 0.8930618762969971)

            finally:
                _remove_files(model_name + "_droplet", max_epochs)
                print(rmse_val, rmse_test)

        elif idx == 1:
            try:
                assert np.isclose(rmse_val[-1], rmse_test)
                assert np.isclose(rmse_val[0], 16.44867706298828)
                assert np.isclose(rmse_val[-1], 3.080655097961426)
            finally:
                _remove_files(model_name + "_droplet", max_epochs)
                print(rmse_val, rmse_test)

        elif idx == 2:
            try:
                assert np.isclose(rmse_val[-1], rmse_test)
                assert np.isclose(rmse_val[0], 11.113712310791016)
                assert np.isclose(rmse_val[-1], 1.0161025524139404)
            finally:
                _remove_files(model_name + "_droplet", max_epochs)
                print(rmse_val, rmse_test)
        model._reset_parameters()
        del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_tweak_parameters_droplet_with_CompartimentedAlchemicalANI():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
    )
    from ..ani import CompartimentedAlchemicalANI2x

    names = ["molDWRow_298"]
    env = "droplet"
    diameter = 10
    max_epochs = 3

    model = CompartimentedAlchemicalANI2x
    model_name = "CompartimentedAlchemicalANI2x"
    # model._reset_parameters()
    model._reset_parameters()

    (rmse_val, rmse_test,) = setup_and_perform_parameter_retraining_with_test_set_split(
        env=env,
        names=names,
        ANImodel=model,
        batch_size=1,
        max_snapshots_per_window=10,
        checkpoint_filename=f"{model_name}_droplet.pt",
        data_path=f"./data/test_data/{env}",
        nr_of_nn=8,
        max_epochs=max_epochs,
        diameter=diameter,
        load_checkpoint=False,
        load_pickled_FEC=False,
    )

    try:
        assert np.isclose(rmse_val[-1], rmse_test)
        assert np.isclose(rmse_val[0], 16.44867706298828)
        assert np.isclose(rmse_val[-1], 3.080704689025879)
    finally:
        _remove_files(model_name + "_droplet", max_epochs)
        print(rmse_val, rmse_test)
    del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_tweak_parameters_droplet_with_CompartimentedAlchemicalANI_load_FEC():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
    )
    from ..ani import CompartimentedAlchemicalANI2x

    names = ["molDWRow_298"]
    env = "droplet"
    diameter = 10
    max_epochs = 3

    ################################
    # standard training run for weights and bias

    model = CompartimentedAlchemicalANI2x
    model_name = "CompartimentedAlchemicalANI2x"
    model._reset_parameters()

    (rmse_val, rmse_test,) = setup_and_perform_parameter_retraining_with_test_set_split(
        env=env,
        names=names,
        ANImodel=model,
        batch_size=1,
        max_snapshots_per_window=10,
        checkpoint_filename=f"{model_name}_droplet.pt",
        data_path=f"./data/test_data/{env}",
        nr_of_nn=8,
        max_epochs=max_epochs,
        diameter=diameter,
        load_checkpoint=False,
        load_pickled_FEC=True,
    )

    try:
        assert np.isclose(rmse_val[-1], rmse_test)
        assert np.isclose(rmse_val[0], 16.44867706298828)
        assert np.isclose(rmse_val[-1], 3.080704689025879)
    finally:
        _remove_files(model_name + "_droplet", max_epochs)
        print(rmse_val, rmse_test)
    model._reset_parameters()
    del model

    #############################################
    # tweak larning rate

    model = CompartimentedAlchemicalANI2x
    model_name = "CompartimentedAlchemicalANI2x"
    model._reset_parameters()

    (rmse_val, rmse_test,) = setup_and_perform_parameter_retraining_with_test_set_split(
        env=env,
        names=names,
        ANImodel=model,
        batch_size=1,
        max_snapshots_per_window=10,
        checkpoint_filename=f"{model_name}_droplet.pt",
        data_path=f"./data/test_data/{env}",
        nr_of_nn=8,
        max_epochs=max_epochs,
        diameter=diameter,
        load_checkpoint=False,
        load_pickled_FEC=True,
        lr_AdamW=1e-4,
        lr_SGD=1e-4,
        weight_decay=0.000001,
    )

    try:
        assert np.isclose(rmse_val[-1], rmse_test)
        assert np.isclose(rmse_val[0], 16.44867706298828)
        assert np.isclose(rmse_val[-1], 14.533025741577148)
    finally:
        _remove_files(model_name + "_droplet", max_epochs)
        print(rmse_val, rmse_test)
    model._reset_parameters()
    del model

    #############################################
    # tweak weight_decay

    model = CompartimentedAlchemicalANI2x
    model_name = "CompartimentedAlchemicalANI2x"
    model._reset_parameters()

    (rmse_val, rmse_test,) = setup_and_perform_parameter_retraining_with_test_set_split(
        env=env,
        names=names,
        ANImodel=model,
        batch_size=1,
        max_snapshots_per_window=10,
        checkpoint_filename=f"{model_name}_droplet.pt",
        data_path=f"./data/test_data/{env}",
        nr_of_nn=8,
        max_epochs=max_epochs,
        diameter=diameter,
        load_checkpoint=False,
        load_pickled_FEC=True,
        lr_AdamW=1e-3,
        lr_SGD=1e-3,
        weight_decay=0.0,
    )

    try:
        assert np.isclose(rmse_val[-1], rmse_test)
        assert np.isclose(rmse_val[0], 16.44867706298828)
        assert np.isclose(rmse_val[-1], 3.080704689025879)
    finally:
        _remove_files(model_name + "_droplet", max_epochs)
        print(rmse_val, rmse_test)
    model._reset_parameters()
    del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_tweak_parameters_droplet_with_AlchemicalANI2x():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
    )
    from ..ani import AlchemicalANI2x

    names = ["molDWRow_298"]
    env = "droplet"
    diameter = 10
    max_epochs = 3

    model = AlchemicalANI2x
    model_name = "AlchemicalANI2x"
    model._reset_parameters()

    (rmse_val, rmse_test,) = setup_and_perform_parameter_retraining_with_test_set_split(
        env=env,
        names=names,
        ANImodel=model,
        batch_size=1,
        max_snapshots_per_window=10,
        checkpoint_filename=f"{model_name}_droplet.pt",
        data_path=f"./data/test_data/{env}",
        nr_of_nn=8,
        max_epochs=max_epochs,
        diameter=diameter,
        load_checkpoint=False,
        load_pickled_FEC=False,
    )

    try:
        assert np.isclose(rmse_val[-1], rmse_test)
        assert np.isclose(rmse_val[0], 16.44867706298828)
        assert np.isclose(rmse_val[-1], 3.080655097961426)
    finally:
        _remove_files(model_name + "_droplet", max_epochs)
        print(rmse_val, rmse_test)
    model._reset_parameters()
    del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow calculation."
)
def test_parameter_gradient():
    from ..constants import kT
    from tqdm import tqdm
    from ..parameter_gradients import FreeEnergyCalculator
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x

    # nr of steps
    #################
    n_steps = 40
    #################

    # specify the system you want to simulate
    name = "molDWRow_298"  # Experimental free energy difference: 1.132369 kcal/mol
    for model in [AlchemicalANI1ccx, AlchemicalANI2x, AlchemicalANI1x]:
        model._reset_parameters()

        (
            energy_function,
            tautomer,
            flipped,
        ) = setup_alchemical_system_and_energy_function(
            name, env="vacuum", ANImodel=model
        )
        x0 = tautomer.get_hybrid_coordinates()
        potential_energy_trajs = []
        ani_trajs = []
        lambdas = np.linspace(0, 1, 5)

        for lamb in tqdm(lambdas):
            # minimize coordinates with a given lambda value
            x0, e_history = energy_function.minimize(x0, maxiter=5, lambda_value=lamb)
            # define energy function with a given lambda value
            energy_and_force = lambda x: energy_function.calculate_force(x, lamb)
            # define langevin object with a given energy function
            langevin = neutromeratio.LangevinDynamics(
                atoms=tautomer.hybrid_atoms, energy_and_force=energy_and_force
            )

            # sampling
            equilibrium_samples, energies, restraint_energies = langevin.run_dynamics(
                x0, n_steps=n_steps, stepsize=1.0 * unit.femtosecond, progress_bar=False
            )
            potential_energy_trajs.append(energies)

            ani_trajs.append(
                md.Trajectory(
                    [x[0] / unit.nanometer for x in equilibrium_samples],
                    tautomer.hybrid_topology,
                )
            )

        # calculate free energy in kT
        fec = FreeEnergyCalculator(
            ani_model=energy_function,
            md_trajs=ani_trajs,
            potential_energy_trajs=potential_energy_trajs,
            lambdas=lambdas,
            bulk_energy_calculation=False,
            max_snapshots_per_window=10,
        )

        # BEWARE HERE: I change the sign of the result since if flipped is TRUE I have
        # swapped tautomer 1 and 2 to mutate from the tautomer WITH the stereobond to the
        # one without the stereobond
        if flipped:
            deltaF = fec._compute_free_energy_difference() * -1
        else:
            deltaF = fec._compute_free_energy_difference()
        print(
            f"Free energy difference {(deltaF.item() * kT).value_in_unit(unit.kilocalorie_per_mole)} kcal/mol"
        )

        deltaF.backward()  # no errors or warnings
        params = list(energy_function.model.optimized_neural_network.parameters())
        none_counter = 0
        for p in params:
            print(p.grad)
            if p.grad == None:  # some are None!
                none_counter += 1

        if not (len(params) == 256 or len(params) == 448):
            raise RuntimeError()
        if not (none_counter == 64 or none_counter == 256):
            raise RuntimeError()
        model._reset_parameters()
        del fec
        del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_parameter_gradient_opt_script():
    import neutromeratio

    env = "vacuum"
    elements = "CHON"
    data_path = f"./data/test_data/{env}"
    for model_name in ["ANI1ccx", "ANI2x"]:

        max_snapshots_per_window = 10
        print(f"Max nr of snapshots: {max_snapshots_per_window}")

        if model_name == "ANI2x":
            model = neutromeratio.ani.AlchemicalANI2x
            print(f"Using {model_name}.")
        elif model_name == "ANI1ccx":
            model = neutromeratio.ani.AlchemicalANI1ccx
            print(f"Using {model_name}.")
        elif model_name == "ANI1x":
            model = neutromeratio.ani.AlchemicalANI1x
            print(f"Using {model_name}.")
        else:
            raise RuntimeError(f"Unknown model name: {model_name}")

        model._reset_parameters()

        if env == "droplet":
            bulk_energy_calculation = False
        else:
            bulk_energy_calculation = True

        names = ["molDWRow_298", "SAMPLmol2", "SAMPLmol4"]

        (
            rmse_validation,
            rmse_test,
        ) = neutromeratio.parameter_gradients.setup_and_perform_parameter_retraining_with_test_set_split(
            env=env,
            ANImodel=model,
            batch_size=1,
            max_snapshots_per_window=max_snapshots_per_window,
            checkpoint_filename=f"parameters_{model_name}_{env}.pt",
            data_path=data_path,
            nr_of_nn=8,
            bulk_energy_calculation=bulk_energy_calculation,
            elements=elements,
            max_epochs=5,
            names=names,
            diameter=10,
            load_checkpoint=False,
            load_pickled_FEC=False,
        )

        model._reset_parameters()
        del model
