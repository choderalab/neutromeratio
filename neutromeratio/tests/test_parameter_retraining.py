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
from neutromeratio.parameter_gradients import chunks


def test_chunks():
    s = [v for v in range(100)]
    # get 10x10 elements
    it = chunks(s, 10)

    elements = next(it)
    len(elements) == 10

    it = chunks(s, 3)
    elements = next(it)
    len(elements) == 3


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Can't upload necessary files."
)
def test_u_ln_50_snapshots():
    from ..parameter_gradients import (
        setup_FEC,
    )
    from ..ani import CompartimentedAlchemicalANI2x
    from neutromeratio.constants import initialize_NUM_PROC

    initialize_NUM_PROC(1)

    # with pickled tautomer object
    name = "SAMPLmol2"
    model, model_name = CompartimentedAlchemicalANI2x, "CompartimentedAlchemicalANI2x"
    model._reset_parameters()
    model_instance = model([0, 0])
    model_instance.load_nn_parameters(
        parameter_path="data/test_data/AlchemicalANI2x_3.pt"
    )
    # vacuum
    fec = setup_FEC(
        name,
        env="vacuum",
        data_path="data/test_data/vacuum",
        ANImodel=model,
        bulk_energy_calculation=True,
        max_snapshots_per_window=50,
        load_pickled_FEC=True,
        include_restraint_energy_contribution=False,
    )
    fec._compute_free_energy_difference()
    # compare to manually scaling
    f_per_molecule = fec.mae_between_potentials_for_snapshots(env="vacuum")
    f_per_atom = fec.mae_between_potentials_for_snapshots(normalized=True, env="vacuum")
    f_scaled_to_mol = (f_per_atom / 100) * len(fec.ani_model.species[0])
    assert np.isclose(f_per_molecule.item(), f_scaled_to_mol.item())
    # for droplet
    # compare to manually scaling
    f_per_molecule = fec.mae_between_potentials_for_snapshots(env="droplet")
    f_per_atom = fec.mae_between_potentials_for_snapshots(
        normalized=True, env="droplet"
    )
    f_scaled_to_mol = (f_per_atom / 100) * len(fec.ani_model.species[0])
    assert np.isclose(f_per_molecule.item(), f_scaled_to_mol.item())


def test_scaling_factor():
    from ..parameter_gradients import _scale_factor_dE, PenaltyFunction

    # linear scaling
    f = _scale_factor_dE(PenaltyFunction(5, 1, 20, 2.0, True), 2)
    assert f.item() == 0.0
    f = _scale_factor_dE(PenaltyFunction(5, 1, 20, 2.0, True), 5)
    assert f.item() == 0.0
    f = _scale_factor_dE(PenaltyFunction(5, 1, 20, 2.0, True), 6)
    assert f.item() == 0.05
    f = _scale_factor_dE(PenaltyFunction(5, 1, 20, 2.0, True), 10)
    assert f.item() == 0.25
    f = _scale_factor_dE(PenaltyFunction(5, 1, 20, 2.0, True), 40)
    assert f.item() == 1.75
    f = _scale_factor_dE(PenaltyFunction(5, 1, 20, 2.0, True), 100)
    assert f.item() == 2.0

    # exp scaling
    f = _scale_factor_dE(PenaltyFunction(5, 2, 20, 2.0, True), 2)
    assert f.item() == 0.0
    f = _scale_factor_dE(PenaltyFunction(5, 2, 20, 2.0, True), 5)
    assert f.item() == 0.0
    f = _scale_factor_dE(PenaltyFunction(5, 2, 20, 2.0, True), 6)
    assert np.isclose(f.item(), 0.0025, rtol=1e-4)
    f = _scale_factor_dE(PenaltyFunction(5, 2, 20, 2.0, True), 10)
    assert np.isclose(f.item(), 0.0625, rtol=1e-4)
    f = _scale_factor_dE(PenaltyFunction(5, 2, 20, 2.0, True), 20)
    assert np.isclose(f.item(), 0.5625, rtol=1e-4)
    f = _scale_factor_dE(PenaltyFunction(5, 2, 20, 2.0, True), 30)
    assert np.isclose(f.item(), 1.5625, rtol=1e-4)
    f = _scale_factor_dE(PenaltyFunction(5, 2, 20, 2.0, True), 40)
    assert np.isclose(f.item(), 2.0, rtol=1e-4)

    f = _scale_factor_dE(
        PenaltyFunction(5, 2, 20, 2.0, True, dE_offset=1.0, dG_offset=1.0), 40
    )
    assert np.isclose(f.item(), 3.0, rtol=1e-4)
    f = _scale_factor_dE(
        PenaltyFunction(5, 1, 20, 2.0, True, dE_offset=1.0, dG_offset=1.0), 100
    )
    assert f.item() == 3.0


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
    for i in range(0, max_epochs):
        try:
            os.remove(f"{name}_{i}.pt")
        except FileNotFoundError:
            pass
    try:
        os.remove(f"{name}_best.pt")
    except FileNotFoundError:
        pass


def test_splitting_function():
    # test that splitting function works as inteded to generate test/training/validation set
    from ..parameter_gradients import _split_names_in_training_validation_test_set

    # initialize set with 100 elements
    s = [i for i in range(100)]

    # perform 60:20:20 split for training:validation:test set
    (
        training_set,
        validation_set,
        test_set,
    ) = _split_names_in_training_validation_test_set(s, 0.2, 0.2)

    assert len(training_set) == 60
    assert len(validation_set) == 20
    assert len(test_set) == 20

    everything = training_set + validation_set + test_set
    # make sure that everything has no duplicates
    assert len(set(everything)) == len(everything)

    # perform 80:10:10 split for training:validation:test set
    (
        training_set,
        validation_set,
        test_set,
    ) = _split_names_in_training_validation_test_set(s, 0.1, 0.1)

    assert len(training_set) == 80
    assert len(validation_set) == 10
    assert len(test_set) == 10

    everything = training_set + validation_set + test_set
    # make sure that everything has no duplicates
    assert len(set(everything)) == len(everything)

    # perform 80:10:10 split for training:validation:test set
    (
        training_set,
        validation_set,
        test_set,
    ) = _split_names_in_training_validation_test_set(s, 0.6, 0.2)

    assert len(training_set) == 20
    assert len(validation_set) == 20
    assert len(test_set) == 60

    everything = training_set + validation_set + test_set
    # make sure that everything has no duplicates
    assert len(set(everything)) == len(everything)


def test_splitting_on_names():
    from ..constants import _get_names
    from ..parameter_gradients import _split_names_in_training_validation_test_set

    # test 80:20:20 split
    test_size = 0.2
    validation_size = 0.2
    # split in training/validation/test set
    # get names of molecules we want to optimize
    names_list = _get_names()
    (
        names_training,
        names_validating,
        names_test,
    ) = _split_names_in_training_validation_test_set(
        names_list, test_size=test_size, validation_size=validation_size
    )

    # save the split for this particular training/validation/test split
    split = {}
    for name, which_set in zip(
        names_training + names_validating + names_test,
        ["training"] * len(names_training)
        + ["validating"] * len(names_validating)
        + ["testing"] * len(names_test),
    ):
        split[name] = which_set

    assert len(names_training) == 212
    assert len(names_validating) == 71
    assert len(names_test) == 71

    # test 20:20:80 split
    test_size = 0.6
    validation_size = 0.2
    # split in training/validation/test set
    # get names of molecules we want to optimize
    names_list = _get_names()
    (
        names_training,
        names_validating,
        names_test,
    ) = _split_names_in_training_validation_test_set(
        names_list, test_size=test_size, validation_size=validation_size
    )

    assert len(names_training) == 70
    assert len(names_validating) == 71
    assert len(names_test) == 213


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_postprocessing_vacuum():
    from ..parameter_gradients import (
        get_perturbed_free_energy_difference,
        get_experimental_values,
    )
    from ..parameter_gradients import setup_FEC
    from ..ani import (
        AlchemicalANI1ccx,
        AlchemicalANI1x,
        AlchemicalANI2x,
        CompartimentedAlchemicalANI2x,
    )
    import numpy as np

    env = "vacuum"
    names = ["molDWRow_298", "SAMPLmol2", "SAMPLmol4"]

    for idx, model in enumerate(
        [
            AlchemicalANI1ccx,
            AlchemicalANI1x,
            AlchemicalANI2x,
            CompartimentedAlchemicalANI2x,
        ]
    ):

        model._reset_parameters()
        print(model.name)
        fec_list = [
            setup_FEC(
                name,
                ANImodel=model,
                env=env,
                data_path="data/test_data/vacuum",
                bulk_energy_calculation=True,
                max_snapshots_per_window=80,
                load_pickled_FEC=False,
                include_restraint_energy_contribution=True,
                save_pickled_FEC=False,
            )
            for name in names
        ]

        # get calc free energy
        f = torch.stack(
            [
                get_perturbed_free_energy_difference(fec).free_energy_estimate
                for fec in fec_list
            ]
        )
        # get exp free energy
        e = torch.stack([get_experimental_values(name) for name in names])
        assert len(f) == 3

        rmse = torch.sqrt(torch.mean((f - e) ** 2))
        print([e._end_state_free_energy_difference[0] for e in fec_list])

        if idx == 0:
            for fec, e2 in zip(
                fec_list, [-1.2104192392489894, -5.31605397264069, 4.055934972298076]
            ):
                assert np.isclose(
                    fec._end_state_free_energy_difference[0], e2, rtol=1e-3
                )
            assert np.isclose(rmse.item(), 5.393606768321977, rtol=1e-3)

        elif idx == 1:
            for fec, e2 in zip(
                fec_list, [-10.201508376053313, -9.919852168528479, 0.6758425107641388]
            ):
                assert np.isclose(
                    fec._end_state_free_energy_difference[0], e2, rtol=1e-3
                )
            assert np.isclose(rmse.item(), 5.464364003709803, rtol=1e-3)

        elif idx == 2:
            for fec, e2 in zip(
                fec_list, [-8.715161329082854, -9.287343875860726, 4.194619951649713]
            ):
                assert np.isclose(
                    fec._end_state_free_energy_difference[0], e2, rtol=1e-3
                )
            assert np.isclose(rmse.item(), 6.115326307713618, rtol=1e-3)

        elif idx == 3:
            for fec, e2 in zip(
                fec_list, [-8.715161329082854, -9.287343875860726, 4.194619951649713]
            ):
                assert np.isclose(
                    fec._end_state_free_energy_difference[0], e2, rtol=1e-3
                )
            assert np.isclose(rmse.item(), 6.115326307713618, rtol=1e-3)

        del model
    del fec_list


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_postprocessing_droplet():
    from ..parameter_gradients import (
        get_perturbed_free_energy_difference,
        get_experimental_values,
    )
    from ..parameter_gradients import setup_FEC
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x

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
        # get calc free energy
        f = torch.stack(
            [
                get_perturbed_free_energy_difference(fec).free_energy_estimate
                for fec in fec_list
            ]
        )
        # get exp free energy
        e = torch.stack([get_experimental_values(name) for name in names])

        if idx == 0:
            print(fec_list)
            assert len(fec_list) == 1

            rmse = torch.sqrt(torch.mean((f - e) ** 2))
            assert np.isclose(
                fec_list[0]._end_state_free_energy_difference[0].item(),
                -1.664279359190441,
                rtol=1e-3,
            )
            assert np.isclose(rmse.item(), 0.23514418557176664, rtol=1e-3)

        elif idx == 1:
            print(fec_list)
            assert len(fec_list) == 1
            rmse = torch.sqrt(torch.mean((f - e) ** 2))
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
        # get calc free energy
        f = torch.stack(
            [
                get_perturbed_free_energy_difference(fec).free_energy_estimate
                for fec in fec_list
            ]
        )
        # get exp free energy
        e = torch.stack([get_experimental_values(name) for name in names])

        if idx == 0:
            print(fec_list)
            assert len(fec_list) == 1
            rmse = torch.sqrt(torch.mean((f - e) ** 2))
            assert np.isclose(
                fec_list[0]._end_state_free_energy_difference[0].item(),
                -1.664279359190441,
                rtol=1e-3,
            )
            assert np.isclose(rmse.item(), 0.23514418557176664, rtol=1e-3)

        elif idx == 1:
            print(fec_list)
            assert len(fec_list) == 1
            rmse = torch.sqrt(torch.mean((f - e) ** 2))
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

        # get calc free energy
        f = torch.stack(
            [
                get_perturbed_free_energy_difference(fec).free_energy_estimate
                for fec in fec_list
            ]
        )
        # get exp free energy
        e = torch.stack([get_experimental_values(name) for name in names])

        if idx == 0:
            print(fec_list)
            assert len(fec_list) == 1
            rmse = torch.sqrt(torch.mean((f - e) ** 2))
            assert np.isclose(
                fec_list[0]._end_state_free_energy_difference[0].item(),
                -1.664279359190441,
                rtol=1e-3,
            )
            assert np.isclose(rmse.item(), 0.23514418557176664, rtol=1e-3)

        elif idx == 1:
            print(fec_list)
            assert len(fec_list) == 1
            rmse = torch.sqrt(torch.mean((f - e) ** 2))
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

        # get calc free energy
        f = torch.stack(
            [
                get_perturbed_free_energy_difference(fec).free_energy_estimate
                for fec in fec_list
            ]
        )
        # get exp free energy
        e = torch.stack([get_experimental_values(name) for name in names])

        if idx == 0:
            print(fec_list)
            assert len(fec_list) == 1
            rmse = torch.sqrt(torch.mean((f - e) ** 2))
            assert np.isclose(
                fec_list[0]._end_state_free_energy_difference[0].item(),
                -1.664279359190441,
                rtol=1e-3,
            )
            assert np.isclose(rmse.item(), 0.23514418557176664, rtol=1e-3)

        elif idx == 1:
            print(fec_list)
            assert len(fec_list) == 1
            rmse = torch.sqrt(torch.mean((f - e) ** 2))
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
def test_snapshot_energy_loss_with_CompartimentedAlchemicalANI2x():
    # test the setup mbar function with different models, environments and potentials
    from ..parameter_gradients import (
        setup_FEC,
        get_perturbed_free_energy_difference,
    )
    from ..ani import CompartimentedAlchemicalANI2x

    # CompartimentedAlchemicalANI2x._reset_parameters()
    name = "molDWRow_298"
    model_instance = CompartimentedAlchemicalANI2x([0, 0])
    env = "vacuum"

    # vacuum
    fec = setup_FEC(
        name,
        env=env,
        diameter=-1,
        data_path="data/test_data/vacuum",
        ANImodel=CompartimentedAlchemicalANI2x,
        bulk_energy_calculation=True,
        max_snapshots_per_window=100,
        load_pickled_FEC=False,
        include_restraint_energy_contribution=False,
        save_pickled_FEC=False,
    )
    assert np.isclose(
        7.981540,
        get_perturbed_free_energy_difference(fec).free_energy_estimate.item(),
        rtol=1e-3,
    )

    assert np.isclose(fec.rmse_between_potentials_for_snapshots().item(), 0.0)

    # load parameters
    model_instance.load_nn_parameters(f"data/test_data/AlchemicalANI2x_3.pt")

    assert np.isclose(
        -11.25832,
        get_perturbed_free_energy_difference(fec).free_energy_estimate.item(),
        rtol=1e-3,
    )
    assert np.isclose(
        fec.rmse_between_potentials_for_snapshots().item(),
        31.581332998622738,
        rtol=1e-3,
    )

    assert np.isclose(
        fec.mae_between_potentials_for_snapshots().item(), 30.506691846410845
    )


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_tweak_parameters_and_class_nn_AlchemicalANI():
    # the tweaked parameters are stored as class variables
    # this can lead to some tricky situations.
    # It also means that whenever any optimization is performed,
    # every new instance of the class has the new parameters
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining,
        PenaltyFunction,
    )
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
            snapshot_penalty_f=PenaltyFunction(5, 1, 20, 1.0, False),
            max_snapshots_per_window=50,
            batch_size=1,
            data_path="./data/test_data/vacuum",
            load_checkpoint=False,
            max_epochs=max_epochs,
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


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_tweak_parameters_and_class_nn_CompartimentedAlchemicalANI():
    # the tweaked parameters are stored as class variables
    # this can lead to some tricky situations.
    # It also means that whenever any optimization is performed,
    # every new instance of the class has the new parameters
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining,
        PenaltyFunction,
    )
    from ..ani import CompartimentedAlchemicalANI2x, CompartimentedAlchemicalANI1ccx

    for model, model_name in [
        (CompartimentedAlchemicalANI2x, "CompartimentedAlchemicalANI2x"),
        (CompartimentedAlchemicalANI1ccx, "CompartimentedAlchemicalANI1ccx"),
    ]:
        model._reset_parameters()
        print(model_name)
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
            snapshot_penalty_f=PenaltyFunction(5, 1, 20, 1.0, False),
            batch_size=1,
            data_path="./data/test_data/vacuum",
            load_checkpoint=False,
            max_epochs=max_epochs,
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
def test_retrain_parameters_vacuum_variable_batch_size_and_n_proc():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
    )
    from ..ani import AlchemicalANI1ccx

    from ..constants import initialize_NUM_PROC

    initialize_NUM_PROC(1)

    names = ["molDWRow_298", "SAMPLmol2", "SAMPLmol4"]
    max_epochs = 4
    model = AlchemicalANI1ccx
    model_name = "AlchemicalANI1ccx"

    # calculate with batch_size=1
    batch_size = 3
    model._reset_parameters()
    (rmse_val, rmse_test) = setup_and_perform_parameter_retraining_with_test_set_split(
        env="vacuum",
        checkpoint_filename=f"{model_name}_vacuum.pt",
        names=names,
        ANImodel=model,
        batch_size=batch_size,
        data_path="./data/test_data/vacuum",
        max_snapshots_per_window=50,
        max_epochs=max_epochs,
        load_checkpoint=False,
    )

    print(rmse_val)
    try:
        assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
        assert np.isclose(rmse_val[0], 5.3938140869140625, rtol=1e-3)
        assert np.isclose(rmse_val[-1], 2.09907078, rtol=1e-3)
    finally:
        _remove_files(model_name + "_vacuum", max_epochs)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_retrain_parameters_vacuum_batch_size():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
    )
    from ..ani import AlchemicalANI1ccx

    from ..constants import initialize_NUM_PROC

    initialize_NUM_PROC(1)

    names = ["molDWRow_298", "SAMPLmol2", "SAMPLmol4"]
    max_epochs = 4
    model = AlchemicalANI1ccx
    model_name = "AlchemicalANI1ccx"

    # calculate with batch_size=1
    # since the names are shuffeled in each training loop this is not deterministic
    batch_size = 1
    model._reset_parameters()
    (rmse_val, rmse_test) = setup_and_perform_parameter_retraining_with_test_set_split(
        env="vacuum",
        checkpoint_filename=f"{model_name}_vacuum.pt",
        names=names,
        ANImodel=model,
        batch_size=batch_size,
        data_path="./data/test_data/vacuum",
        max_snapshots_per_window=50,
        max_epochs=max_epochs,
        load_checkpoint=False,
    )

    print(rmse_val)
    try:
        assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
        assert np.isclose(rmse_val[0], 5.3938140869140625, rtol=1e-3)
        # training performance for batch_size=1 is not deterministic
        # assert np.isclose(rmse_val[-1], 1.824, rtol=1e-3)
    finally:
        _remove_files(model_name + "_vacuum", max_epochs)

    # calculate with batch_size=3
    batch_size = 3
    model._reset_parameters()
    (rmse_val, rmse_test) = setup_and_perform_parameter_retraining_with_test_set_split(
        env="vacuum",
        checkpoint_filename=f"{model_name}_vacuum.pt",
        names=names,
        ANImodel=model,
        batch_size=batch_size,
        data_path="./data/test_data/vacuum",
        max_snapshots_per_window=50,
        max_epochs=max_epochs,
        load_checkpoint=False,
    )

    print(rmse_val)
    try:
        assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
        assert np.isclose(rmse_val[0], 5.393814086, rtol=1e-3)
        assert np.isclose(rmse_val[-1], 2.09907078, rtol=1e-3)
    finally:
        _remove_files(model_name + "_vacuum", max_epochs)

    # calculate with batch_size=4
    model._reset_parameters()
    batch_size = 4
    (rmse_val, rmse_test) = setup_and_perform_parameter_retraining_with_test_set_split(
        env="vacuum",
        checkpoint_filename=f"{model_name}_vacuum.pt",
        names=names,
        ANImodel=model,
        batch_size=batch_size,
        data_path="./data/test_data/vacuum",
        max_snapshots_per_window=50,
        max_epochs=max_epochs,
        load_checkpoint=False,
    )

    print(rmse_val)
    try:
        assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
        assert np.isclose(rmse_val[0], 5.393814086, rtol=1e-3)
        assert np.isclose(rmse_val[-1], 2.09907078, rtol=1e-3)
    finally:
        _remove_files(model_name + "_vacuum", max_epochs)

    # calculate with batch_size=2
    batch_size = 2
    model._reset_parameters()
    (rmse_val, rmse_test) = setup_and_perform_parameter_retraining_with_test_set_split(
        env="vacuum",
        checkpoint_filename=f"{model_name}_vacuum.pt",
        names=names,
        ANImodel=model,
        batch_size=batch_size,
        data_path="./data/test_data/vacuum",
        max_snapshots_per_window=50,
        max_epochs=max_epochs,
        load_checkpoint=False,
    )

    print(rmse_val)
    try:
        assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
        assert np.isclose(rmse_val[0], 5.393814086, rtol=1e-3)
        # not deterministic
        # assert np.isclose(rmse_val[-1], 2.858044385, rtol=1e-3)
    finally:
        _remove_files(model_name + "_vacuum", max_epochs)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_retrain_parameters_vacuum_batch_size_all_potentials():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
    )
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x

    # calculate with batch_size=1
    names = ["molDWRow_298", "SAMPLmol2", "SAMPLmol4"]
    max_epochs = 4
    # calculate with batch_size=1
    batch_size = 1
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
            batch_size=batch_size,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=50,
            max_epochs=max_epochs,
            load_checkpoint=False,
        )

        if idx == 0:
            print(rmse_val)
            try:
                assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
                assert np.isclose(rmse_val[0], 5.3938140869140625, rtol=1e-3)
                # assert np.isclose(rmse_val[-1], 1.8240348100662231, rtol=1e-3)
            finally:
                _remove_files(model_name + "_vacuum", max_epochs)

        if idx == 1:
            print(rmse_val)
            try:

                assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
                assert np.isclose(rmse_val[0], 5.187891006469727, rtol=1e-3)
                # assert np.isclose(rmse_val[-1], 3.470416307449341, rtol=1e-3)
            finally:
                _remove_files(model_name + "_vacuum", max_epochs)

        if idx == 2:
            print(rmse_val)
            try:
                assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
                assert np.isclose(rmse_val[0], 4.582426071166992, rtol=1e-3)
                # assert np.isclose(rmse_val[-1], 2.3228771686553955, rtol=1e-3)
            finally:
                _remove_files(model_name + "_vacuum", max_epochs)
        model._reset_parameters()

    # calculate with batch_size=3
    batch_size = 3
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
            batch_size=batch_size,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=50,
            max_epochs=max_epochs,
            load_checkpoint=False,
        )

        if idx == 0:
            print(rmse_val)
            try:
                assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
                assert np.isclose(rmse_val[0], 5.3938140869140625, rtol=1e-3)
                assert np.isclose(rmse_val[-1], 2.098975658416748, rtol=1e-3)
            finally:
                _remove_files(model_name + "_vacuum", max_epochs)

        if idx == 1:
            print(rmse_val)
            try:

                assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
                assert np.isclose(rmse_val[0], 5.187891006469727, rtol=1e-3)
                assert np.isclose(rmse_val[-1], 2.672308921813965, rtol=1e-3)
            finally:
                _remove_files(model_name + "_vacuum", max_epochs)

        if idx == 2:
            print(rmse_val)
            try:
                assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
                assert np.isclose(rmse_val[0], 4.582426071166992, rtol=1e-3)
                assert np.isclose(rmse_val[-1], 2.2336010932922363, rtol=1e-3)
            finally:
                _remove_files(model_name + "_vacuum", max_epochs)
        model._reset_parameters()


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
            max_epochs=max_epochs,
            load_checkpoint=False,
        )

        print(rmse_val)
        try:
            assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
            assert np.isclose(rmse_val[0], 5.7811503410339355, rtol=1e-3)
            assert np.isclose(rmse_val[-1], 2.1603381633758545, rtol=1e-3)
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
            max_epochs=max_epochs,
            load_checkpoint=False,
        )

        print(rmse_val)
        try:
            assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
            assert np.isclose(rmse_val[0], 5.7811503410339355, rtol=1e-3)
            assert np.isclose(rmse_val[-1], 2.1603381633758545, rtol=1e-3)
        finally:
            _remove_files(model_name + "_vacuum", max_epochs)

        model._reset_parameters()
        del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_retrain_energy_penalty():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
        PenaltyFunction,
    )
    from ..ani import CompartimentedAlchemicalANI2x
    from neutromeratio.constants import initialize_NUM_PROC

    initialize_NUM_PROC(1)

    # without pickled tautomer object
    names = ["molDWRow_298", "SAMPLmol2", "SAMPLmol4"]
    max_epochs = 20
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
            batch_size=3,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=50,
            max_epochs=max_epochs,
            load_checkpoint=False,
            lr_AdamW=1e-5,
            lr_SGD=1e-5,
            snapshot_penalty_f=PenaltyFunction(5, 1, 20, 1.0, True),
        )

        print(rmse_val)
        try:
            assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
            assert np.isclose(rmse_val[0], 5.187891006469727, rtol=1e-3)
            assert np.isclose(
                rmse_val[-1], 4.32212495803833, rtol=1e-1
            )  # NOTE: This is not zero!
        finally:
            _remove_files(model_name + "_vacuum", max_epochs)

        model._reset_parameters()
        del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_retrain_mp_mp1():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
    )
    from ..ani import CompartimentedAlchemicalANI2x
    from neutromeratio.constants import initialize_NUM_PROC

    initialize_NUM_PROC(2)

    # without pickled tautomer object
    names = ["molDWRow_298", "SAMPLmol2", "SAMPLmol4"]
    max_epochs = 20
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
            batch_size=3,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=50,
            max_epochs=max_epochs,
            load_checkpoint=False,
            lr_AdamW=1e-5,
            lr_SGD=1e-5,
        )

        print(rmse_val)
        try:
            assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
            assert np.isclose(rmse_val[0], 5.187891006469727, rtol=1e-3)
            assert np.isclose(
                rmse_val[-1], 4.2210774421691895, rtol=1e-3
            )  # NOTE: This is not zero!
        finally:
            _remove_files(model_name + "_vacuum", max_epochs)

        model._reset_parameters()
        del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_retrain_mp_mp2():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
    )
    from ..ani import CompartimentedAlchemicalANI2x
    from neutromeratio.constants import initialize_NUM_PROC

    # initialize_NUM_PROC(2)

    # without pickled tautomer object
    names = ["molDWRow_298", "SAMPLmol2", "SAMPLmol4"]
    max_epochs = 20
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
            batch_size=3,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=50,
            max_epochs=max_epochs,
            load_checkpoint=False,
            lr_AdamW=1e-5,
            lr_SGD=1e-5,
        )

        print(rmse_val)
        try:
            assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
            assert np.isclose(rmse_val[0], 5.187891006469727, rtol=1e-3)
            assert np.isclose(
                rmse_val[-1], 4.2210774421691895, rtol=1e-3
            )  # NOTE: This is not zero!
        finally:
            _remove_files(model_name + "_vacuum", max_epochs)

        model._reset_parameters()
        del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_retrain_mp_mp3_epoch20():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
    )
    from ..ani import CompartimentedAlchemicalANI2x
    from neutromeratio.constants import initialize_NUM_PROC

    # initialize_NUM_PROC(3)

    # without pickled tautomer object
    names = ["molDWRow_298", "SAMPLmol2", "SAMPLmol4"]
    max_epochs = 20
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
            batch_size=3,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=50,
            max_epochs=max_epochs,
            load_checkpoint=False,
            lr_AdamW=1e-5,
            lr_SGD=1e-5,
        )

        print(rmse_val)
        try:
            assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
            assert np.isclose(rmse_val[0], 5.187891006469727, rtol=1e-3)
            assert np.isclose(
                rmse_val[-1], 4.2210774421691895, rtol=1e-3
            )  # NOTE: This is not zero!
        finally:
            _remove_files(model_name + "_vacuum", max_epochs)

        model._reset_parameters()
        del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_retrain_mp_mp3_epoch50():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
    )
    from ..ani import CompartimentedAlchemicalANI2x
    from neutromeratio.constants import initialize_NUM_PROC

    # initialize_NUM_PROC(3)

    # without pickled tautomer object
    names = ["molDWRow_298", "SAMPLmol2", "SAMPLmol4"]
    max_epochs = 50
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
            batch_size=3,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=50,
            max_epochs=max_epochs,
            load_checkpoint=False,
            lr_AdamW=1e-5,
            lr_SGD=1e-5,
        )

        print(rmse_val)
        try:
            assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
            assert np.isclose(rmse_val[0], 5.187891006469727, rtol=1e-3)
            assert np.isclose(
                rmse_val[-1], 3.07578, rtol=1e-2
            )  # NOTE: This is not zero!
        finally:
            _remove_files(model_name + "_vacuum", max_epochs)

        model._reset_parameters()
        del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_timing():
    from ..ani import CompartimentedAlchemicalANI2x
    from ..parameter_gradients import (
        get_unperturbed_free_energy_difference,
        get_perturbed_free_energy_difference,
    )
    from neutromeratio.parameter_gradients import setup_FEC, calculate_mse
    import time

    name = "molDWRow_298"  # , "SAMPLmol2", "SAMPLmol4"]
    model = CompartimentedAlchemicalANI2x
    env = "vacuum"
    fec = setup_FEC(
        name,
        ANImodel=model,
        env=env,
        bulk_energy_calculation=True,
        data_path="./data/test_data/vacuum",
        max_snapshots_per_window=150,
        load_pickled_FEC=True,
        include_restraint_energy_contribution=False,
        save_pickled_FEC=False,
    )

    start = time.time()
    f = get_perturbed_free_energy_difference(fec)
    end = time.time()
    print(end - start)

    start = time.time()
    g = get_unperturbed_free_energy_difference(fec)
    end = time.time()
    print(end - start)

    r = calculate_mse(f, f)
    print(r)

    start = time.time()
    r.backward()
    end = time.time()
    print(f"Time: {end - start}")


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_retrain_parameters_CompartimentedAlchemicalANI2x_for_50_epochs():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
    )
    from ..ani import CompartimentedAlchemicalANI2x

    # without pickled tautomer object
    names = ["molDWRow_298"]
    max_epochs = 50
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
            max_snapshots_per_window=150,
            max_epochs=max_epochs,
            load_checkpoint=False,
            lr_AdamW=1e-5,
            lr_SGD=1e-5,
        )

        print(rmse_val)
        try:
            assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
            assert np.isclose(rmse_val[0], 6.82616662979126, rtol=1e-3)
            assert np.isclose(
                rmse_val[-1], 2.5604844093322754, rtol=1e-2
            )  # NOTE: This is not zero!
        finally:
            _remove_files(model_name + "_vacuum", max_epochs)

        model._reset_parameters()
        del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_retrain_parameters_CompartimentedAlchemicalANI2x_including_dE_single_tautomer_pair():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
        PenaltyFunction,
    )
    from ..ani import CompartimentedAlchemicalANI2x

    # without pickled tautomer object
    names = ["molDWRow_298"]
    max_epochs = 50
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
            snapshot_penalty_f=PenaltyFunction(1, 1, 1, 1, True),
            batch_size=1,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=150,
            max_epochs=max_epochs,
            load_checkpoint=False,
            lr_AdamW=1e-5,
            lr_SGD=1e-5,
        )

        print(rmse_val)
        try:
            assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
            assert np.isclose(rmse_val[0], 6.82616662979126, rtol=1e-3)
            assert np.isclose(
                rmse_val[-1], 3.3715248107910156, rtol=1e-2
            )  # NOTE: This is not zero!
        finally:
            _remove_files(model_name + "_vacuum", max_epochs)

        model._reset_parameters()
        del model


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_retrain_parameters_CompartimentedAlchemicalANI2x_extended_loss_20_epochs_three_tautomers():
    from ..parameter_gradients import (
        setup_and_perform_parameter_retraining_with_test_set_split,
    )
    from ..ani import CompartimentedAlchemicalANI2x

    # without pickled tautomer object
    names = ["molDWRow_298", "SAMPLmol2", "SAMPLmol4"]
    max_epochs = 20
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
            batch_size=3,
            data_path="./data/test_data/vacuum",
            max_snapshots_per_window=200,
            max_epochs=max_epochs,
            load_checkpoint=False,
            lr_AdamW=1e-5,
            lr_SGD=1e-5,
        )

        print(rmse_val)
        try:
            assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
            assert np.isclose(rmse_val[0], 5.62937593460083, rtol=1e-3)
            assert np.isclose(
                rmse_val[-1], 4.591619491577148, rtol=1e-2
            )  # NOTE: This is not zero!
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
            max_epochs=max_epochs,
            load_checkpoint=False,
        )

        print(rmse_val)
        try:
            assert np.isclose(rmse_val[-1], rmse_test, rtol=1e-3)
            assert np.isclose(rmse_val[0], 5.7811503410339355, rtol=1e-3)
            assert np.isclose(rmse_val[-1], 2.1603381633758545, rtol=1e-3)
        finally:
            _remove_files(model_name + "_vacuum", max_epochs)

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
            max_epochs=max_epochs,
            load_checkpoint=False,
        )

        print(rmse_val)
        try:
            assert np.isclose(rmse_val[-1], rmse_test)
            assert np.isclose(rmse_val[0], 5.7811503410339355)
            assert np.isclose(rmse_val[-1], 2.1603381633758545, rtol=1e-3)
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
            max_epochs=max_epochs,
            diameter=diameter,
            load_checkpoint=False,
        )

        if idx == 0:
            try:
                assert np.isclose(rmse_val[-1], rmse_test)
                assert np.isclose(rmse_val[0], 0.23515522480010986, rtol=1e-3)
                assert np.isclose(rmse_val[-1], 0.8930618762969971, rtol=1e-3)

            finally:
                _remove_files(model_name + "_droplet", max_epochs)
                print(rmse_val, rmse_test)

        elif idx == 1:
            try:
                assert np.isclose(rmse_val[-1], rmse_test)
                assert np.isclose(rmse_val[0], 16.44867706298828)
                assert np.isclose(rmse_val[-1], 3.080655097961426, rtol=1e-3)
            finally:
                _remove_files(model_name + "_droplet", max_epochs)
                print(rmse_val, rmse_test)

        elif idx == 2:
            try:
                assert np.isclose(rmse_val[-1], rmse_test)
                assert np.isclose(rmse_val[0], 11.113712310791016)
                assert np.isclose(rmse_val[-1], 1.0161025524139404, rtol=1e-3)
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
        max_epochs=max_epochs,
        diameter=diameter,
        load_checkpoint=False,
    )

    try:
        assert np.isclose(rmse_val[-1], rmse_test)
        assert np.isclose(rmse_val[0], 16.44867706298828)
        assert np.isclose(rmse_val[-1], 3.080704689025879, rtol=1e-3)
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
        max_epochs=max_epochs,
        diameter=diameter,
        load_checkpoint=False,
    )

    try:
        assert np.isclose(rmse_val[-1], rmse_test)
        assert np.isclose(rmse_val[0], 16.44867706298828)
        assert np.isclose(rmse_val[-1], 3.080704689025879, rtol=1e-3)
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
        max_epochs=max_epochs,
        diameter=diameter,
        load_checkpoint=False,
        lr_AdamW=1e-4,
        lr_SGD=1e-4,
        weight_decay=0.000001,
    )

    try:
        assert np.isclose(rmse_val[-1], rmse_test)
        assert np.isclose(rmse_val[0], 16.44867706298828)
        assert np.isclose(rmse_val[-1], 14.533025741577148, rtol=1e-3)
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
        max_epochs=max_epochs,
        diameter=diameter,
        load_checkpoint=False,
        lr_AdamW=1e-3,
        lr_SGD=1e-3,
        weight_decay=0.0,
    )

    try:
        assert np.isclose(rmse_val[-1], rmse_test)
        assert np.isclose(rmse_val[0], 16.44867706298828)
        assert np.isclose(rmse_val[-1], 3.080704689025879, rtol=1e-3)
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
        max_epochs=max_epochs,
        diameter=diameter,
        load_checkpoint=False,
    )

    try:
        assert np.isclose(rmse_val[-1], rmse_test)
        assert np.isclose(rmse_val[0], 16.44867706298828)
        assert np.isclose(rmse_val[-1], 3.080655097961426, rtol=1e-3)
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
    os.environ.get("TRAVIS", None) == "true", reason="Can't upload necessary files."
)
def test_parameter_gradient_opt_script():
    import neutromeratio

    env = "vacuum"
    elements = "CHON"
    data_path = f"./data/test_data/{env}"
    for model_name in ["ANI1ccx", "ANI2x"]:

        max_snapshots_per_window = 50
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
            elements=elements,
            max_epochs=5,
            names=names,
            diameter=10,
            load_checkpoint=False,
        )

        model._reset_parameters()
        del model
