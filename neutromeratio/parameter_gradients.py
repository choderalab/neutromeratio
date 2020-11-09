# TODO: gradient of MBAR_estimated free energy difference w.r.t. model parameters

import logging

import numpy as np
import torch
from pymbar import MBAR
from pymbar.timeseries import detectEquilibration
from simtk import unit
from tqdm import tqdm
from glob import glob
from .ani import ANI1_force_and_energy, ANI
from neutromeratio.constants import (
    _get_names,
    hartree_to_kJ_mol,
    device,
    platform,
    kT,
    exclude_set_ANI,
    mols_with_charge,
    multiple_stereobonds,
)
import torchani, torch
import os
import neutromeratio
import pickle
import mdtraj as md
from typing import Tuple

logger = logging.getLogger(__name__)


class FreeEnergyCalculator:
    def __init__(
        self,
        ani_model: ANI1_force_and_energy,
        md_trajs: list,
        bulk_energy_calculation: bool,
        potential_energy_trajs: list,
        lambdas: list,
        max_snapshots_per_window: int = 200,
        pickle_path: str = "",
    ):
        """
        Uses mbar to calculate the free energy difference between trajectories.
        Parameters
        ----------
        ani_model : AlchemicalANI
            model used for energy calculation
        ani_trajs : list
            trajectories
        potential_energy_trajs : list
            energy trace of trajectories
        lambdas : list
            all lambda states
        """

        def get_mix(lambda0, lambda1, lam=0.0):
            return (1 - lam) * lambda0 + lam * lambda1

        K = len(lambdas)
        assert len(md_trajs) == K
        assert len(potential_energy_trajs) == K
        self.ani_model = ani_model
        self.potential_energy_trajs = (
            potential_energy_trajs  # for detecting equilibrium
        )
        self.lambdas = lambdas

        if pickle_path and os.path.isfile((pickle_path)):
            logger.info(f"Loading MBAR results from: {pickle_path}")
            p = pickle.load(open(pickle_path, "rb"))
            self.mbar = p["MBAR"]
            self.coordinates = p["coordinates"]

        else:
            ani_trajs = {}
            for lam, traj, potential_energy in zip(
                self.lambdas, md_trajs, self.potential_energy_trajs
            ):
                # detect equilibrium
                equil, g = detectEquilibration(
                    np.array([e / kT for e in potential_energy])
                )[:2]
                # thinn snapshots and return max_snapshots_per_window confs
                quarter_traj_limit = int(len(traj) / 4)
                snapshots = traj[min(quarter_traj_limit, equil) :].xyz * unit.nanometer
                further_thinning = max(
                    int(len(snapshots) / max_snapshots_per_window), 1
                )
                snapshots = snapshots[::further_thinning][:max_snapshots_per_window]
                ani_trajs[lam] = snapshots
                logger.info(len(snapshots))

                # test that we have a lower number of snapshots than max_snapshots_per_window
                if max_snapshots_per_window == -1:
                    logger.debug(
                        f"There are {len(snapshots)} snapshots per lambda state"
                    )

                if max_snapshots_per_window != -1 and (
                    len(snapshots) > max_snapshots_per_window
                ):
                    raise RuntimeError(
                        f"There are {len(snapshots)} snapshots per lambda state (max: {max_snapshots_per_window}). Aborting."
                    )

                # test that we have not less than 60% of max_snapshots_per_window
                if max_snapshots_per_window != -1 and len(snapshots) < (
                    int(max_snapshots_per_window * 0.6)
                ):
                    raise RuntimeError(
                        f"There are only {len(snapshots)} snapshots per lambda state. Aborting."
                    )
                # test that we have not less than 40 snapshots
                if len(snapshots) < 40:
                    logger.critical(
                        f"There are only {len(snapshots)} snapshots per lambda state. Be careful."
                    )

            snapshots = []
            N_k = []
            for lam in sorted(self.lambdas):
                logger.debug(f"lamb: {lam}")
                N_k.append(len(ani_trajs[lam]))
                snapshots.extend(ani_trajs[lam])
                logger.debug(f"Snapshots per lambda {lam}: {len(ani_trajs[lam])}")

            if len(snapshots) < 300:
                logger.critical(
                    f"Total number of snapshots is {len(snapshots)} -- is this enough?"
                )

            coordinates = [
                sample / unit.angstrom for sample in snapshots
            ] * unit.angstrom

            logger.debug(f"len(coordinates): {len(coordinates)}")

            # end-point energies
            if bulk_energy_calculation:
                lambda0_e = self.ani_model.calculate_energy(
                    coordinates,
                    lambda_value=0.0,
                    original_neural_network=True,
                    requires_grad_wrt_coordinates=False,
                    requires_grad_wrt_parameters=False,
                ).energy
                lambda1_e = self.ani_model.calculate_energy(
                    coordinates,
                    lambda_value=1.0,
                    original_neural_network=True,
                    requires_grad_wrt_coordinates=False,
                    requires_grad_wrt_parameters=False,
                ).energy
            else:
                lambda0_e = []
                lambda1_e = []
                for coord in coordinates:
                    # getting coord from [N][3] to [1][N][3]
                    coord = np.array([coord / unit.angstrom]) * unit.angstrom
                    e0 = self.ani_model.calculate_energy(
                        coord,
                        lambda_value=0.0,
                        original_neural_network=True,
                        requires_grad_wrt_coordinates=False,
                        requires_grad_wrt_parameters=False,
                    ).energy
                    lambda0_e.append(e0[0] / kT)
                    e1 = self.ani_model.calculate_energy(
                        coord,
                        lambda_value=1.0,
                        original_neural_network=True,
                        requires_grad_wrt_coordinates=False,
                        requires_grad_wrt_parameters=False,
                    ).energy
                    lambda1_e.append(e1[0] / kT)
                lambda0_e = np.array(lambda0_e) * kT
                lambda1_e = np.array(lambda1_e) * kT

            logger.debug(f"len(lambda0_e): {len(lambda0_e)}")

            u_kn = np.stack(
                [
                    get_mix(lambda0_e / kT, lambda1_e / kT, lam)
                    for lam in sorted(self.lambdas)
                ]
            )

            del lambda0_e
            del lambda1_e

            self.mbar = MBAR(u_kn, N_k)
            self.coordinates = coordinates

            if pickle_path:
                # only save if pickle_path is defined
                p = {"MBAR": self.mbar, "coordinates": self.coordinates}
                pickle.dump(p, open(pickle_path, "wb"))
                logger.info(f"Saving MBAR results to: {pickle_path}")

    @property
    def free_energy_differences(self):
        """matrix of free energy differences"""
        return self.mbar.getFreeEnergyDifferences(return_dict=True)["Delta_f"]

    @property
    def free_energy_difference_uncertainties(self):
        """matrix of asymptotic uncertainty-estimates accompanying free energy differences"""
        return self.mbar.getFreeEnergyDifferences(return_dict=True)["dDelta_f"]

    @property
    def _end_state_free_energy_difference(self):
        """DeltaF[lambda=1 --> lambda=0]"""
        results = self.mbar.getFreeEnergyDifferences(return_dict=True)
        return results["Delta_f"][0, -1], results["dDelta_f"][0, -1]

    def _compute_perturbed_free_energies(self, u_ln):
        """compute perturbed free energies at new thermodynamic states l"""
        assert type(u_ln) == torch.Tensor

        # Note: Use mbar estimate with orginal parameter set!
        N_k = torch.tensor(
            self.mbar.N_k, dtype=torch.double, requires_grad=False, device=device
        )
        f_k = torch.tensor(
            self.mbar.f_k, dtype=torch.double, requires_grad=False, device=device
        )
        u_kn = torch.tensor(
            self.mbar.u_kn, dtype=torch.double, requires_grad=False, device=device
        )

        # importance weighting
        log_q_k = f_k - u_kn.T
        A = log_q_k + torch.log(N_k)
        log_denominator_n = torch.logsumexp(A, dim=1)

        B = -u_ln - log_denominator_n
        return -torch.logsumexp(B, dim=1)

    def _form_u_ln(self):

        # bring list of unit'd coordinates in [N][K][3] * unit shape
        coordinates = self.coordinates

        # Note: Use class neural network here (that might or might not be modified)!
        u_0 = self.ani_model.calculate_energy(
            coordinates,
            lambda_value=0.0,
            original_neural_network=False,
            requires_grad_wrt_coordinates=False,
        ).energy_tensor

        # Note: Use class neural network here (that might or might not be modified)!
        u_1 = self.ani_model.calculate_energy(
            coordinates,
            lambda_value=1.0,
            original_neural_network=False,
            requires_grad_wrt_coordinates=False,
        ).energy_tensor

        u_ln = torch.stack([u_0, u_1])
        return u_ln

    def _compute_free_energy_difference(self):
        u_ln = self._form_u_ln()
        f_k = self._compute_perturbed_free_energies(u_ln)
        return f_k[1] - f_k[0]


def torchify(x):
    return torch.tensor(x, dtype=torch.double, requires_grad=True, device=device)


def get_perturbed_free_energy_difference(fec_list: list) -> torch.Tensor:
    """
    Gets a list of fec instances and returns a torch.tensor with
    the computed free energy differences.

    Arguments:
        fec_list {list[torch.tensor]}

    Returns:
        torch.tensor -- calculated free energy in kT
    """
    calc = []

    for idx, fec in enumerate(fec_list):
        if fec.flipped:
            deltaF = fec._compute_free_energy_difference() * -1.0
        else:
            deltaF = fec._compute_free_energy_difference()
        calc.append(deltaF)
    return torch.stack([e for e in calc])


def get_unperturbed_free_energy_difference(fec_list: list):
    """
    Gets a list of fec instances and returns a torch.tensor with
    the computed free energy differences.

    Arguments:
        fec_list {list[torch.tensor]}

    Returns:
        torch.tensor -- calculated free energy in kT
    """
    calc = []

    for idx, fec in enumerate(fec_list):
        if fec.flipped:
            deltaF = fec._end_state_free_energy_difference[0] * -1.0
        else:
            deltaF = fec._end_state_free_energy_difference[0]
        calc.append(deltaF)
    logger.debug(calc)
    return torch.stack([torchify(e) for e in calc])


def get_experimental_values(names: list) -> torch.Tensor:
    """
    Returns the experimental free energy differen in solution for the tautomer pair

    Returns:
        [torch.Tensor] -- experimental free energy in kT
    """
    from neutromeratio.analysis import _get_exp_results

    exp_results = _get_exp_results()
    exp = []

    for idx, name in enumerate(names):
        e_in_kT = (exp_results[name]["energy"] * unit.kilocalorie_per_mole) / kT
        exp.append(e_in_kT)
    logger.debug(exp)
    return torchify(exp)


def calculate_rmse_between_exp_and_calc(
    names: list,
    model: ANI,
    data_path: str,
    bulk_energy_calculation: bool,
    env: str,
    max_snapshots_per_window: int,
    perturbed_free_energy: bool = True,
    diameter: int = -1,
    load_pickled_tautomer_object: bool = False,
) -> Tuple[float, list]:
    """
    Returns the RMSE between calculated and experimental free energy differences as float.

    Arguments:
        names {list} -- list of system names considered for RMSE calculation
        data_path {str} -- data path to the location of the trajectories
        thinning {int} -- nth frame considerd
        max_snapshots_per_window {int} -- maximum number of snapshots per window

    Returns:
        [float] -- returns the RMSE as float
        [list] -- return a list of the calculated dG values
    """
    if env == "droplet" and diameter == -1:
        raise RuntimeError(
            f"Something went wrong. Diameter is set for {diameter}. Aborting."
        )

    e_calc, e_exp = [], []
    it = tqdm(names)

    for name in it:
        fec_list = [
            setup_mbar(
                name=name,
                ANImodel=model,
                env=env,
                bulk_energy_calculation=bulk_energy_calculation,
                data_path=data_path,
                max_snapshots_per_window=max_snapshots_per_window,
                diameter=diameter,
                load_pickled_tautomer_object=load_pickled_tautomer_object,
            )
        ]

        # append calculated values
        if perturbed_free_energy:
            e_calc.append(get_perturbed_free_energy_difference(fec_list)[0].item())
        else:
            e_calc.append(get_unperturbed_free_energy_difference(fec_list)[0].item())

        # append experimental values
        e_exp.append(get_experimental_values([name])[0].item())
        current_rmse = calculate_rmse(
            torch.tensor(e_calc, device=device), torch.tensor(e_exp, device=device)
        ).item()

        it.set_description(f"RMSE: {current_rmse}")

        if current_rmse > 50:
            logger.critical(f"RMSE above 50 with {current_rmse}: {name}")
            logger.critical(names)

    return current_rmse, e_calc


def calculate_mse(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    assert t1.size() == t2.size()
    return torch.mean((t1 - t2) ** 2)


def calculate_rmse(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    assert t1.size() == t2.size()
    return torch.sqrt(calculate_mse(t1, t2))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _split_names_in_training_validation_test_set(
    names_list: list,
) -> Tuple[list, list, list]:
    from sklearn.model_selection import train_test_split

    names_training_validating, names_test = train_test_split(names_list, test_size=0.2)
    print(
        f"Len of training/validation set: {len(names_training_validating)}/{len(names_list)}"
    )

    names_training, names_validating = train_test_split(
        names_training_validating, test_size=0.2
    )
    print(
        f"Len of training set: {len(names_training)}/{len(names_training_validating)}"
    )
    print(
        f"Len of validating set: {len(names_validating)}/{len(names_training_validating)}"
    )
    print(f"Len of test set: {len(names_test)}/{len(names_list)}")

    return names_training, names_validating, names_test


def setup_and_perform_parameter_retraining_with_test_set_split(
    ANImodel: ANI,
    env: str,
    max_snapshots_per_window: int,
    checkpoint_filename: str,
    diameter: int = -1,
    batch_size: int = 10,
    elements: str = "CHON",
    data_path: str = "../data/",
    nr_of_nn: int = 8,
    max_epochs: int = 10,
    bulk_energy_calculation: bool = True,
    load_checkpoint: bool = True,
    names: list = [],
    load_pickled_tautomer_object: bool = True,
) -> Tuple[list, float]:

    """
    Calculates the free energy of a staged free energy simulation,
    tweaks the neural net parameter so that using reweighting the difference
    between the experimental and calculated free energy is minimized.

    Parameters:
        ANImode [ANI]: The ANI model object (not instance)
        emv [str]: either 'vacuum' or droplet
        max_snapshots_per_window [int]: how many snapshots should be considered per lambda state
        checkpoint_filename [str]: filename to save checkpoint files
        diameter [int] [opt]: diameter in Angstrom for droplet
        batch_size [int] [opt]
        elements [str] [opt]:


    Raises:
        RuntimeError: [description]

    Returns:
        [type]: [description]
    """

    import random

    assert int(nr_of_nn) <= 8 and int(nr_of_nn) >= 1

    if env == "droplet" and diameter == -1:
        raise RuntimeError(f"Did you forget to pass the 'diamter' argument? Aborting.")

    if names:
        logger.critical(
            "BE CAREFUL! This is not a real training run but a test run with user specified molecule names."
        )
        logger.critical("Validating and test set are the same")
        names_training = names
        names_validating = names
        names_test = names
    else:
        # split in training/validation/test set
        # get names of molecules we want to optimize
        names_list = _get_names()
        (
            names_training,
            names_validating,
            names_test,
        ) = _split_names_in_training_validation_test_set(names_list)

    # rmsd calculation on test set
    rmse_test, dG_calc_test_initial = calculate_rmse_between_exp_and_calc(
        model=ANImodel,
        names=names_test,
        diameter=diameter,
        data_path=data_path,
        bulk_energy_calculation=bulk_energy_calculation,
        env=env,
        max_snapshots_per_window=max_snapshots_per_window,
        load_pickled_tautomer_object=load_pickled_tautomer_object,
    )

    print(f"RMSE on test set BEFORE optimization: {rmse_test}")

    split = {}
    for name, which_set in zip(
        names_training + names_validating + names_test,
        ["training"] * len(names_training)
        + ["validation"] * len(names_validating)
        + ["testing"] * len(names_validating),
    ):
        split[name] = which_set
    pickle.dump(split, open(f"training_validation_tests.pickle", "wb+"))

    # save batch loss through epochs
    rmse_validation = setup_and_perform_parameter_retraining(
        ANImodel=ANImodel,
        env=env,
        checkpoint_filename=checkpoint_filename,
        max_snapshots_per_window=max_snapshots_per_window,
        diameter=diameter,
        batch_size=batch_size,
        data_path=data_path,
        nr_of_nn=nr_of_nn,
        max_epochs=max_epochs,
        elements=elements,
        load_checkpoint=load_checkpoint,
        bulk_energy_calculation=bulk_energy_calculation,
        names_training=names_training,
        names_validating=names_validating,
        load_pickled_tautomer_object=load_pickled_tautomer_object,
    )

    # final rmsd calculation on test set
    rmse_test, dG_calc_test_final = calculate_rmse_between_exp_and_calc(
        model=ANImodel,
        names=names_test,
        diameter=diameter,
        data_path=data_path,
        bulk_energy_calculation=bulk_energy_calculation,
        env=env,
        max_snapshots_per_window=max_snapshots_per_window,
        load_pickled_tautomer_object=load_pickled_tautomer_object,
    )
    print(f"RMSE on test set AFTER optimization: {rmse_test}")

    # write out data on dG for test set after optimization
    exp_results = neutromeratio.analysis._get_exp_results()

    results = {}
    for name, e_initial, e_final in zip(
        names_test, dG_calc_test_initial, dG_calc_test_final
    ):
        results[name] = (
            e_initial,
            e_final,
            (exp_results[name]["energy"] * unit.kilocalorie_per_mole) / kT,
        )
    pickle.dump(results, open(f"results_for_test_set.pickle", "wb+"))

    return rmse_validation, rmse_test


def _save_checkpoint(
    model, AdamW, AdamW_scheduler, SGD, SGD_scheduler, latest_checkpoint
):
    torch.save(
        {
            "nn": model.tweaked_neural_network.state_dict(),
            "AdamW": AdamW.state_dict(),
            "SGD": SGD.state_dict(),
            "AdamW_scheduler": AdamW_scheduler.state_dict(),
            "SGD_scheduler": SGD_scheduler.state_dict(),
        },
        latest_checkpoint,
    )


def _perform_training(
    ANImodel: ANI,
    nr_of_nn: int,
    names_training: list,
    names_validating: list,
    checkpoint_filename: str,
    max_epochs: int,
    elements: str,
    bulk_energy_calculation: bool,
    env: str,
    diameter: int,
    batch_size: int,
    data_path: str,
    max_snapshots_per_window: int,
    load_checkpoint: bool,
    rmse_validation: list,
    load_pickled_tautomer_object: bool,
) -> list:

    early_stopping_learning_rate = 1.0e-5
    AdamW, AdamW_scheduler, SGD, SGD_scheduler = _get_nn_layers(
        nr_of_nn, ANImodel, elements=elements
    )

    if load_checkpoint:
        _load_checkpoint(
            checkpoint_filename, ANImodel, AdamW, AdamW_scheduler, SGD, SGD_scheduler
        )

    logger.info(f"training starting from epoch {AdamW_scheduler.last_epoch + 1}")

    base = checkpoint_filename.split(".")[0]
    best_model_checkpoint = f"{base}_best.pt"

    ## training loop
    for i in range(AdamW_scheduler.last_epoch + 1, max_epochs):

        # get the learning group
        learning_rate = AdamW.param_groups[0]["lr"]
        if learning_rate < early_stopping_learning_rate:
            break

        # checkpoint -- if best parameters on validation set save parameters
        if AdamW_scheduler.is_better(rmse_validation[-1], AdamW_scheduler.best):
            torch.save(
                ANImodel.tweaked_neural_network.state_dict(), best_model_checkpoint
            )

        # define the stepsize
        AdamW_scheduler.step(rmse_validation[-1])
        SGD_scheduler.step(rmse_validation[-1])

        # perform the parameter optimization and importance weighting
        _tweak_parameters(
            names_training=names_training,
            ANImodel=ANImodel,
            AdamW=AdamW,
            SGD=SGD,
            env=env,
            bulk_energy_calculation=bulk_energy_calculation,
            diameter=diameter,
            batch_size=batch_size,
            data_path=data_path,
            max_snapshots_per_window=max_snapshots_per_window,
            load_pickled_tautomer_object=load_pickled_tautomer_object,
        )

        # calculate the new free energies on the validation set with optimized parameters
        current_rmse, dG_calc_validation = calculate_rmse_between_exp_and_calc(
            names_validating,
            model=ANImodel,
            diameter=diameter,
            data_path=data_path,
            bulk_energy_calculation=bulk_energy_calculation,
            env=env,
            max_snapshots_per_window=max_snapshots_per_window,
            perturbed_free_energy=True,
            load_pickled_tautomer_object=load_pickled_tautomer_object,
        )

        rmse_validation.append(current_rmse)

        print(
            f"RMSE on validation set: {rmse_validation[-1]} at epoch {AdamW_scheduler.last_epoch}"
        )

        _save_checkpoint(
            ANImodel,
            AdamW,
            AdamW_scheduler,
            SGD,
            SGD_scheduler,
            f"{base}_{AdamW_scheduler.last_epoch}.pt",
        )

    # _save_checkpoint(
    #     ANImodel, AdamW, AdamW_scheduler, SGD, SGD_scheduler, checkpoint_filename
    # )

    return rmse_validation


def _tweak_parameters(
    names_training: list,
    ANImodel: ANI,
    AdamW,
    SGD,
    env: str,
    diameter: int,
    bulk_energy_calculation: bool,
    batch_size: int,
    data_path: str,
    max_snapshots_per_window: int,
    load_pickled_tautomer_object: bool,
):
    """
    _tweak_parameters

    Parameters
    ----------
    names_training : list
        names used for training
    ANImodel : ANI
        the ANI class used for energy calculation
    AdamW :
        AdamW instance
    SGD : [type]
        SGD instance
    env : str
        either 'droplet' or vacuum
    diameter : int
        diameter of droplet or -1
    bulk_energy_calculation : bool
        controls if the energy calculation should be performed in bluk (parallel) or sequential
    batch_size : int
        the number of molecules used to calculate the loss
    data_path : str
        the location where the trajectories are saved
    max_snapshots_per_window : int
        the number of snapshots per lambda state to consider

    """

    # iterate over batches of molecules
    it = tqdm(chunks(names_training, batch_size))

    for idx, names in enumerate(it):

        # define setup_mbar function
        # get mbar instances in a list
        fec_list = [
            setup_mbar(
                name=name,
                ANImodel=ANImodel,
                env=env,
                data_path=data_path,
                bulk_energy_calculation=bulk_energy_calculation,
                max_snapshots_per_window=max_snapshots_per_window,
                diameter=diameter,
                load_pickled_tautomer_object=load_pickled_tautomer_object,
            )
            for name in names
        ]

        # calculate the free energies
        calc_free_energy_difference = get_perturbed_free_energy_difference(fec_list)
        # obtain the experimental free energies
        exp_free_energy_difference = get_experimental_values(names)
        # calculate the loss as MSE
        loss = calculate_mse(calc_free_energy_difference, exp_free_energy_difference)
        it.set_description(f"Batch {idx} -- MSE: {loss.item()}")

        # optimization steps
        AdamW.zero_grad()
        SGD.zero_grad()
        loss.backward()
        AdamW.step()
        SGD.step()

        del calc_free_energy_difference


def _load_checkpoint(
    latest_checkpoint, model, AdamW, AdamW_scheduler, SGD, SGD_scheduler
):
    # save checkpoint
    if os.path.isfile(latest_checkpoint):
        checkpoint = torch.load(latest_checkpoint)
        model.tweaked_neural_network.load_state_dict(checkpoint["nn"])
        AdamW.load_state_dict(checkpoint["AdamW"])
        SGD.load_state_dict(checkpoint["SGD"])
        AdamW_scheduler.load_state_dict(checkpoint["AdamW_scheduler"])
        SGD_scheduler.load_state_dict(checkpoint["SGD_scheduler"])
    else:
        logger.info(f"Checkoint {latest_checkpoint} does not exist.")


def _get_nn_layers(nr_of_nn: int, ANImodel: ANI, elements: str = "CHON"):

    if elements == "CHON":
        logger.info("Using `CHON` elements.")
        weight_layers, bias_layers = _get_nn_layers_CHON(nr_of_nn, ANImodel)
    elif elements == "CN":
        logger.info("Using `CN` elements.")
        weight_layers, bias_layers = _get_nn_layers_CN(nr_of_nn, ANImodel)
    elif elements == "H":
        logger.info("Using `H` elements.")
        weight_layers, bias_layers = _get_nn_layers_H(nr_of_nn, ANImodel)
    elif elements == "C":
        logger.info("Using `C` elements.")
        weight_layers, bias_layers = _get_nn_layers_C(nr_of_nn, ANImodel)
    else:
        raise RuntimeError("Only `CHON` or `CN` as atoms allowed. Aborting.")

    # set up minimizer for weights
    AdamW = torch.optim.AdamW(weight_layers)
    # set up minimizer for bias
    SGD = torch.optim.SGD(bias_layers, lr=1e-3)

    AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        AdamW, factor=0.5, patience=100, threshold=0
    )
    SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        SGD, factor=0.5, patience=100, threshold=0
    )

    return (AdamW, AdamW_scheduler, SGD, SGD_scheduler)


def _get_nn_layers_C(nr_of_nn: int, ANImodel: ANI):
    weight_layers = []
    bias_layers = []
    layer = 6
    model = ANImodel.tweaked_neural_network

    for nn in model[:nr_of_nn]:
        weight_layers.extend(
            [
                {"params": [nn.C[layer].weight], "weight_decay": 0.000001},
            ]
        )
        bias_layers.extend(
            [
                {"params": [nn.C[layer].bias]},
            ]
        )
    return (weight_layers, bias_layers)


def _get_nn_layers_H(nr_of_nn: int, ANImodel: ANI):
    weight_layers = []
    bias_layers = []
    layer = 6
    model = ANImodel.tweaked_neural_network

    for nn in model[:nr_of_nn]:
        weight_layers.extend(
            [
                {"params": [nn.H[layer].weight], "weight_decay": 0.000001},
            ]
        )
        bias_layers.extend(
            [
                {"params": [nn.H[layer].bias]},
            ]
        )
    return (weight_layers, bias_layers)


def _get_nn_layers_CN(nr_of_nn: int, ANImodel: ANI):
    weight_layers = []
    bias_layers = []
    layer = 6
    model = ANImodel.tweaked_neural_network

    for nn in model[:nr_of_nn]:
        weight_layers.extend(
            [
                {"params": [nn.C[layer].weight], "weight_decay": 0.000001},
                {"params": [nn.N[layer].weight], "weight_decay": 0.000001},
            ]
        )
        bias_layers.extend(
            [
                {"params": [nn.C[layer].bias]},
                {"params": [nn.N[layer].bias]},
            ]
        )
    return (weight_layers, bias_layers)


def _get_nn_layers_CHON(nr_of_nn: int, ANImodel: ANI):
    weight_layers = []
    bias_layers = []
    model = ANImodel.tweaked_neural_network
    layer = 6
    for nn in model[:nr_of_nn]:
        weight_layers.extend(
            [
                {"params": [nn.C[layer].weight], "weight_decay": 0.000001},
                {"params": [nn.H[layer].weight], "weight_decay": 0.000001},
                {"params": [nn.O[layer].weight], "weight_decay": 0.000001},
                {"params": [nn.N[layer].weight], "weight_decay": 0.000001},
            ]
        )
        bias_layers.extend(
            [
                {"params": [nn.C[layer].bias]},
                {"params": [nn.H[layer].bias]},
                {"params": [nn.O[layer].bias]},
                {"params": [nn.N[layer].bias]},
            ]
        )

    return (weight_layers, bias_layers)


def setup_and_perform_parameter_retraining(
    ANImodel: ANI,
    env: str,
    checkpoint_filename: str,
    max_snapshots_per_window: int,
    diameter: int = -1,
    batch_size: int = 1,
    data_path: str = "../data/",
    nr_of_nn: int = 8,
    max_epochs: int = 10,
    elements: str = "CHON",
    load_checkpoint: bool = True,
    bulk_energy_calculation: bool = True,
    names_training: list = [],
    names_validating: list = [],
    load_pickled_tautomer_object: bool = True,
):
    """
    Much of this code is taken from:
    https://aiqm.github.io/torchani/examples/nnp_training.html
    but instead of training on atomic energies the training is
    performed on relative free energies.

    The function is set up to be called from the notebook or scripts folder.

    Keyword Arguments:
        batch_size {int} -- how many molecules should be used to calculate the MSE
        data_path {str} -- should point to where the dcd files are located (default: {"../data/"})
        nr_of_nn {int} -- number of neural networks that should be tweeked, maximum 8  (default: {8})
        max_epochs {int} -- (default: {10})
        thinning {int} -- nth frame taken from simulation (default: {100})
        max_snapshots_per_window {int} -- total number of frames taken from simulation (default: {100})
        names {list} -- only used for tests -- this loads specific molecules (default: {[]})

    Returns:
        (list, list, float) -- rmse on validation set, rmse on training set, rmse on test set
    """

    assert int(batch_size) <= 10 and int(batch_size) >= 1
    assert int(nr_of_nn) <= 8 and int(nr_of_nn) >= 1

    if load_pickled_tautomer_object:
        model_instance = ANImodel(
            [0, 0]
        )  # NOTE: The model needs a single instance to work with pickled tautomer objects

    if env == "droplet" and diameter == -1:
        raise RuntimeError(f"Did you forget to pass the 'diamter' argument? Aborting.")

    # save batch loss through epochs
    rmse_validation = []

    # calculate the rmse on the current parameters for the validation set
    rmse_validation_set, dG_calc_validation = calculate_rmse_between_exp_and_calc(
        names_validating,
        model=ANImodel,
        data_path=data_path,
        bulk_energy_calculation=bulk_energy_calculation,
        env=env,
        max_snapshots_per_window=max_snapshots_per_window,
        diameter=diameter,
        perturbed_free_energy=False,
        load_pickled_tautomer_object=load_pickled_tautomer_object,
    )

    rmse_validation.append(rmse_validation_set)
    print(f"RMSE on validation set: {rmse_validation[-1]} at first epoch")

    ### main training loop
    rmse_validation = _perform_training(
        ANImodel=ANImodel,
        nr_of_nn=nr_of_nn,
        names_training=names_training,
        names_validating=names_validating,
        rmse_validation=rmse_validation,
        checkpoint_filename=checkpoint_filename,
        max_epochs=max_epochs,
        elements=elements,
        env=env,
        bulk_energy_calculation=bulk_energy_calculation,
        diameter=diameter,
        batch_size=batch_size,
        data_path=data_path,
        load_checkpoint=load_checkpoint,
        max_snapshots_per_window=max_snapshots_per_window,
        load_pickled_tautomer_object=load_pickled_tautomer_object,
    )

    return rmse_validation


def setup_mbar(
    name: str,
    max_snapshots_per_window: int,
    ANImodel: ANI,
    bulk_energy_calculation: bool,
    env: str = "vacuum",
    checkpoint_file: str = "",
    data_path: str = "../data/",
    diameter: int = -1,
    load_pickled_tautomer_object: bool = False,
):

    from neutromeratio.analysis import setup_alchemical_system_and_energy_function
    import os

    data_path = os.path.abspath(data_path)
    if not os.path.exists(data_path):
        raise RuntimeError(f"{data_path} does not exist!")

    tautomer_pickle = f"{data_path}/{name}/{name}_tautomer_system_{max_snapshots_per_window}_for_{ANImodel.name}.pickle"
    if os.path.exists(tautomer_pickle) and load_pickled_tautomer_object:
        t = pickle.load(open(tautomer_pickle, "rb"))
        print(f"{tautomer_pickle} loading ...")
        return t

    if name in exclude_set_ANI + mols_with_charge + multiple_stereobonds:
        raise RuntimeError(
            f"{name} is part of the list of excluded molecules. Aborting"
        )
    if not (env == "vacuum" or env == "droplet"):
        raise RuntimeError("Only keyword vacuum or droplet are allowed as environment.")
    if env == "droplet" and diameter == -1:
        raise RuntimeError("Something went wrong.")

    def parse_lambda_from_dcd_filename(dcd_filename):
        """parsed the dcd filename

        Arguments:
            dcd_filename {str} -- how is the dcd file called?

        Returns:
            [float] -- lambda value
        """
        l = dcd_filename[: dcd_filename.find(f"_energy_in_{env}")].split("_")
        lam = l[-3]
        return float(lam)

    #######################

    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name,
        ANImodel=ANImodel,
        checkpoint_file=checkpoint_file,
        env=env,
        diameter=diameter,
        base_path=f"{data_path}/{name}/",
    )
    # and lambda values in list
    dcds = glob(f"{data_path}/{name}/*.dcd")

    lambdas = []
    md_trajs = []
    energies = []

    # read in all the frames from the trajectories
    if env == "vacuum":
        top = tautomer.hybrid_topology
    else:
        top = f"{data_path}/{name}/{name}_in_droplet.pdb"

    for dcd_filename in dcds:
        lam = parse_lambda_from_dcd_filename(dcd_filename)
        lambdas.append(lam)
        traj = md.load_dcd(dcd_filename, top=top)
        logger.debug(f"Nr of frames in trajectory: {len(traj)}")
        md_trajs.append(traj)
        f = open(
            f"{data_path}/{name}/{name}_lambda_{lam:0.4f}_energy_in_{env}.csv", "r"
        )
        energies.append(np.array([float(e) * kT for e in f]))
        f.close()

    if len(lambdas) < 5:
        raise RuntimeError(f"Below 5 lambda states for {name}")

    assert len(lambdas) == len(energies)
    assert len(lambdas) == len(md_trajs)

    if env == "vacuum":
        pickle_path = f"{data_path}/{name}/{name}_{ANImodel.name}_{max_snapshots_per_window}_{len(tautomer.hybrid_atoms)}_atoms.pickle"
    else:
        pickle_path = f"{data_path}/{name}/{name}_{ANImodel.name}_{max_snapshots_per_window}_{diameter}A_{len(tautomer.ligand_in_water_atoms)}_atoms.pickle"

    # calculate free energy in kT
    fec = FreeEnergyCalculator(
        ani_model=energy_function,
        md_trajs=md_trajs,
        potential_energy_trajs=energies,
        lambdas=lambdas,
        pickle_path=pickle_path,
        bulk_energy_calculation=bulk_energy_calculation,
        max_snapshots_per_window=max_snapshots_per_window,
    )

    fec.flipped = flipped
    pickle.dump(fec, open(tautomer_pickle, "wb+"))
    return fec


def setup_mbar_for_new_tautomer_pairs(
    name: str,
    t1_smiles: str,
    t2_smiles: str,
    max_snapshots_per_window: int,
    ANImodel: ANI,
    bulk_energy_calculation: bool,
    env: str = "vacuum",
    checkpoint_file: str = "",
    data_path: str = "../data/",
    diameter: int = -1,
):

    from neutromeratio.analysis import setup_new_alchemical_system_and_energy_function
    import os

    if not (env == "vacuum" or env == "droplet"):
        raise RuntimeError("Only keyword vacuum or droplet are allowed as environment.")
    if env == "droplet" and diameter == -1:
        raise RuntimeError("Something went wrong.")

    def parse_lambda_from_dcd_filename(dcd_filename):
        """parsed the dcd filename

        Arguments:
            dcd_filename {str} -- how is the dcd file called?

        Returns:
            [float] -- lambda value
        """
        l = dcd_filename[: dcd_filename.find(f"_energy_in_{env}")].split("_")
        lam = l[-3]
        return float(lam)

    data_path = os.path.abspath(data_path)
    if not os.path.exists(data_path):
        raise RuntimeError(f"{data_path} does not exist!")

    #######################
    (energy_function, tautomer,) = setup_new_alchemical_system_and_energy_function(
        name=name,
        t1_smiles=t1_smiles,
        t2_smiles=t2_smiles,
        ANImodel=ANImodel,
        checkpoint_file=checkpoint_file,
        env=env,
        diameter=diameter,
        base_path=f"{data_path}/{name}/",
    )
    # and lambda values in list
    dcds = glob(f"{data_path}/{name}/*.dcd")

    lambdas = []
    md_trajs = []
    energies = []

    # read in all the frames from the trajectories
    if env == "vacuum":
        top = tautomer.hybrid_topology
    else:
        top = f"{data_path}/{name}/{name}_in_droplet.pdb"

    for dcd_filename in dcds:
        lam = parse_lambda_from_dcd_filename(dcd_filename)
        lambdas.append(lam)
        traj = md.load_dcd(dcd_filename, top=top)
        logger.debug(f"Nr of frames in trajectory: {len(traj)}")
        md_trajs.append(traj)
        f = open(
            f"{data_path}/{name}/{name}_lambda_{lam:0.4f}_energy_in_{env}.csv", "r"
        )
        energies.append(np.array([float(e) * kT for e in f]))
        f.close()

    if len(lambdas) < 5:
        raise RuntimeError(f"Below 5 lambda states for {name}")

    assert len(lambdas) == len(energies)
    assert len(lambdas) == len(md_trajs)

    if env == "vacuum":
        pickle_path = f"{data_path}/{name}/{name}_{ANImodel.name}_{max_snapshots_per_window}_{len(tautomer.hybrid_atoms)}_atoms.pickle"
    else:
        pickle_path = f"{data_path}/{name}/{name}_{ANImodel.name}_{max_snapshots_per_window}_{diameter}A_{len(tautomer.ligand_in_water_atoms)}_atoms.pickle"

    # calculate free energy in kT
    fec = FreeEnergyCalculator(
        ani_model=energy_function,
        md_trajs=md_trajs,
        potential_energy_trajs=energies,
        lambdas=lambdas,
        pickle_path=pickle_path,
        bulk_energy_calculation=bulk_energy_calculation,
        max_snapshots_per_window=max_snapshots_per_window,
    )

    return fec
