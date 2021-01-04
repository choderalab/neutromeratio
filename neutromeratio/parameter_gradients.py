# TODO: gradient of MBAR_estimated free energy difference w.r.t. model parameters

import logging

import torch
import numpy as np
from pymbar import MBAR
from pymbar.timeseries import detectEquilibration
from simtk import unit
from torch.types import Number
from tqdm import tqdm
from glob import glob
from .ani import ANI_force_and_energy, ANI
from neutromeratio.constants import _get_names, device, kT
import neutromeratio.constants
import torch
import os
import neutromeratio
import mdtraj as md
from typing import Tuple
import pickle
import torch.multiprocessing as mp
import timeit

logger = logging.getLogger(__name__)


class FreeEnergyCalculator:
    def __init__(
        self,
        ani_model: ANI_force_and_energy,
        md_trajs: list,
        bulk_energy_calculation: bool,
        potential_energy_trajs: list,
        lambdas: list,
        max_snapshots_per_window: int = 200,
        include_restraint_energy_contribution: bool = True,
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
        include_restraint_energy_contribution : bool (default: True)
            including restraint energy
        """

        K = len(lambdas)
        assert len(md_trajs) == K
        assert len(potential_energy_trajs) == K
        assert include_restraint_energy_contribution in (False, True)

        self.ani_model = ani_model
        self.include_restraint_energy_contribution = (
            include_restraint_energy_contribution
        )
        potential_energy_trajs = potential_energy_trajs  # for detecting equilibrium
        self.lambdas = lambdas

        coordinates, N_k = self.combine_snapshots(
            md_trajs, potential_energy_trajs, max_snapshots_per_window
        )
        self.setup_mbar(coordinates, N_k, bulk_energy_calculation)

    def combine_snapshots(
        self,
        md_trajs: list,
        potential_energy_trajs: list,
        max_snapshots_per_window: int,
    ) -> Tuple[list, list]:
        """
        combine_snapshots Merges trajectories to a single list of snapshots

        Parameters
        ----------
        md_trajs : list
            list of mdtraj.Trajectory instances
        potential_energy_trajs : list
            list of potential energies for each snapshot
        max_snapshots_per_window : int
            maximum number of snapshots for each lambda state

        Returns
        -------
        Tuple
            merged snapshots, N_k

        Raises
        ------
        RuntimeError
            is raised if the number of snapshot per lambda is larger than max_snapshots_per_window
        RuntimeError
            is raised if the number of snapshot per lambda is smaller than max_snapshots_per_window * 0.6
        """
        ani_trajs = {}
        for lam, traj, potential_energy in zip(
            self.lambdas, md_trajs, potential_energy_trajs
        ):
            # detect equilibrium
            equil, g = detectEquilibration(
                np.array([e / kT for e in potential_energy])
            )[:2]
            # thinn snapshots and return max_snapshots_per_window confs
            quarter_traj_limit = int(len(traj) / 4)
            snapshots = traj[min(quarter_traj_limit, equil) :].xyz * unit.nanometer
            further_thinning = max(int(len(snapshots) / max_snapshots_per_window), 1)
            snapshots = snapshots[::further_thinning][:max_snapshots_per_window]
            ani_trajs[lam] = snapshots
            logger.info(len(snapshots))

            # test that we have a lower number of snapshots than max_snapshots_per_window
            if max_snapshots_per_window == -1:
                logger.debug(f"There are {len(snapshots)} snapshots per lambda state")

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

        coordinates = [sample / unit.angstrom for sample in snapshots] * unit.angstrom
        return coordinates, N_k

    def setup_mbar(self, coordinates: list, N_k: list, bulk_energy_calculation: bool):

        """
        [summary]
        """

        def get_mix(lambda0, lambda1, lam=0.0):
            return (1 - lam) * lambda0 + lam * lambda1

        logger.debug(f"len(coordinates): {len(coordinates)}")

        # end-point energies
        if bulk_energy_calculation:
            lambda0_e = self.ani_model.calculate_energy(
                coordinates,
                lambda_value=0.0,
                original_neural_network=True,
                requires_grad_wrt_coordinates=False,
                requires_grad_wrt_parameters=False,
                include_restraint_energy_contribution=self.include_restraint_energy_contribution,
            ).energy
            lambda1_e = self.ani_model.calculate_energy(
                coordinates,
                lambda_value=1.0,
                original_neural_network=True,
                requires_grad_wrt_coordinates=False,
                requires_grad_wrt_parameters=False,
                include_restraint_energy_contribution=self.include_restraint_energy_contribution,
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
                    include_restraint_energy_contribution=self.include_restraint_energy_contribution,
                ).energy
                lambda0_e.append(e0[0] / kT)
                e1 = self.ani_model.calculate_energy(
                    coord,
                    lambda_value=1.0,
                    original_neural_network=True,
                    requires_grad_wrt_coordinates=False,
                    requires_grad_wrt_parameters=False,
                    include_restraint_energy_contribution=self.include_restraint_energy_contribution,
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

    @property
    def free_energy_differences(self) -> np.ndarray:
        """matrix of free energy differences"""
        return self.mbar.getFreeEnergyDifferences(return_dict=True)["Delta_f"]

    @property
    def free_energy_difference_uncertainties(self) -> np.ndarray:
        """matrix of asymptotic uncertainty-estimates accompanying free energy differences"""
        return self.mbar.getFreeEnergyDifferences(return_dict=True)["dDelta_f"]

    @property
    def _end_state_free_energy_difference(self) -> np.ndarray:
        """DeltaF[lambda=1 --> lambda=0]"""
        results = self.mbar.getFreeEnergyDifferences(return_dict=True)
        return results["Delta_f"][0, -1], results["dDelta_f"][0, -1]

    def _compute_perturbed_free_energies(self, u_ln) -> torch.Tensor:
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

    def _form_u_ln(self, original_neural_network: bool = False) -> torch.Tensor:

        # bring list of unit'd coordinates in [N][K][3] * unit shape
        coordinates = self.coordinates

        # Note: Use class neural network here (that might or might not be modified)!
        u_0 = self.ani_model.calculate_energy(
            coordinates,
            lambda_value=0.0,
            original_neural_network=original_neural_network,
            requires_grad_wrt_coordinates=False,
            include_restraint_energy_contribution=self.include_restraint_energy_contribution,
        ).energy_tensor

        # Note: Use class neural network here (that might or might not be modified)!
        u_1 = self.ani_model.calculate_energy(
            coordinates,
            lambda_value=1.0,
            original_neural_network=original_neural_network,
            requires_grad_wrt_coordinates=False,
            include_restraint_energy_contribution=self.include_restraint_energy_contribution,
        ).energy_tensor

        u_ln = torch.stack([u_0, u_1])
        return u_ln

    def _compute_free_energy_difference(self) -> torch.Tensor:
        u_ln = self._form_u_ln()
        f_k = self._compute_perturbed_free_energies(u_ln)
        # keep u_ln in memory
        self.u_ln_rho_star_wrt_parameters = u_ln
        return f_k[1] - f_k[0]

    def get_u_ln_for_rho_and_rho_star(self) -> Tuple[(torch.Tensor, torch.Tensor)]:
        u_ln_rho = torch.stack(
            [
                torch.tensor(
                    self.mbar.u_kn[0],  # lambda=0
                    dtype=torch.double,
                    requires_grad=False,
                    device=device,
                ),
                torch.tensor(
                    self.mbar.u_kn[-1],  # lambda=1
                    dtype=torch.double,
                    requires_grad=False,
                    device=device,
                ),
            ]
        )

        if torch.is_tensor(self.u_ln_rho_star_wrt_parameters):
            u_ln_rho_star = self.u_ln_rho_star_wrt_parameters
        else:
            logger.critical(
                "u_ln_rho_star is not set. Calculating. This __might__ indicate a problem!"
            )
            u_ln_rho_star = self._form_u_ln()
        return u_ln_rho, u_ln_rho_star

    def rmse_between_potentials_for_snapshots(self) -> torch.Tensor:
        u_ln_rho, u_ln_rho_star = self.get_u_ln_for_rho_and_rho_star()
        return calculate_rmse(u_ln_rho, u_ln_rho_star)

    def mae_between_potentials_for_snapshots(self) -> torch.Tensor:
        u_ln_rho, u_ln_rho_star = self.get_u_ln_for_rho_and_rho_star()
        return calculate_mae(u_ln_rho, u_ln_rho_star)


def torchify(x):
    return torch.tensor(x, dtype=torch.double, requires_grad=True, device=device)


def get_perturbed_free_energy_difference(fec: FreeEnergyCalculator) -> torch.Tensor:
    """
    Gets a list of fec instances and returns a torch.tensor with
    the computed free energy differences.

    Arguments:
        fec_list {list[torch.tensor]}

    Returns:
        torch.tensor -- calculated free energy in kT
    """
    if fec.flipped:
        deltaF = fec._compute_free_energy_difference() * -1.0
    else:
        deltaF = fec._compute_free_energy_difference()

    return deltaF


def get_unperturbed_free_energy_difference(fec: FreeEnergyCalculator):
    """
    Gets a list of fec instances and returns a torch.tensor with
    the computed free energy differences.

    Arguments:
        fec_list {list[torch.tensor]}

    Returns:
        torch.tensor -- calculated free energy in kT
    """

    if fec.flipped:
        deltaF = fec._end_state_free_energy_difference[0] * -1.0
    else:
        deltaF = fec._end_state_free_energy_difference[0]

    return torchify(deltaF)


def get_experimental_values(name: str) -> torch.Tensor:
    """
    Returns the experimental free energy differen in solution for the tautomer pair

    Returns:
        [torch.Tensor] -- experimental free energy in kT
    """
    from neutromeratio.analysis import _get_exp_results

    exp_results = _get_exp_results()

    e_in_kT = (exp_results[name]["energy"] * unit.kilocalorie_per_mole) / kT
    logger.debug(e_in_kT)
    return torchify(e_in_kT)


def _setup_FEC(prop: dict):
    f = setup_FEC(**prop)
    f.name = prop["name"]
    return f


def calculate_rmse_between_exp_and_calc(
    names: list,
    model: ANI,
    data_path: str,
    bulk_energy_calculation: bool,
    env: str,
    max_snapshots_per_window: int,
    perturbed_free_energy: bool = True,
    diameter: int = -1,
    load_pickled_FEC: bool = False,
    include_restraint_energy_contribution: bool = False,
) -> Tuple[Number, list]:

    """
    calculate_rmse_between_exp_and_calc Returns the RMSE between calculated and experimental free energy differences as float

    Parameters
    ----------
    names : list
        list of system names considered for RMSE calculation
    model : ANI
        [description]
    data_path : str
        data path to the location of the trajectories
    bulk_energy_calculation : bool
        [description]
    env : str
        [description]
    max_snapshots_per_window : int
        maximum number of snapshots per window
    perturbed_free_energy : bool, optional
        [description], by default True
    diameter : int, optional
        [description], by default -1
    load_pickled_FEC : bool, optional
        [description], by default False
    include_restraint_energy_contribution : bool, optional
        [description], by default False

    Returns
    -------
    Tuple[float, list]
        returns the RMSE as float, calculated dG values as list

    Raises
    ------
    RuntimeError
        is raised if diameter is not specified for a droplet system
    """
    if env == "droplet" and diameter == -1:
        raise RuntimeError(
            f"Something went wrong. Diameter is set for {diameter}. Aborting."
        )

    e_calc, e_exp = [], []
    current_rmse = -1.0
    it = tqdm(chunks(names, neutromeratio.constants.NUM_PROC))
    for name_list in it:
        prop_list = [
            {
                "name": name,
                "ANImodel": model,
                "env": env,
                "bulk_energy_calculation": bulk_energy_calculation,
                "data_path": data_path,
                "max_snapshots_per_window": max_snapshots_per_window,
                "diameter": diameter,
                "load_pickled_FEC": load_pickled_FEC,
                "include_restraint_energy_contribution": include_restraint_energy_contribution,
            }
            for name in name_list
        ]

        # only one CPU is specified, mp is not needed
        if neutromeratio.constants.NUM_PROC == 1:
            FEC_list = map(_setup_FEC, prop_list)
        # reading in parallel
        else:
            with mp.Pool(processes=len(name_list)) as pool:
                FEC_list = pool.map(_setup_FEC, prop_list)
                pool.close()
                pool.terminate()

        for fec in FEC_list:
            # append calculated values
            if perturbed_free_energy:
                e_calc.append(get_perturbed_free_energy_difference(fec).item())
            else:
                e_calc.append(get_unperturbed_free_energy_difference(fec).item())

            # append experimental values
            e_exp.append(get_experimental_values(fec.name).item())
            current_rmse = calculate_rmse(
                torch.tensor(e_calc, device=device), torch.tensor(e_exp, device=device)
            ).item()


            if current_rmse > 50:
                logger.critical(f"RMSE above 50 with {current_rmse}: {fec.name}")
                logger.critical(names)
        
        it.set_description(f"RMSE: {current_rmse}")

    return current_rmse, e_calc


def calculate_mae(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    assert t1.size() == t2.size()
    return torch.mean(abs(t1 - t2))


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
    names_list: list, test_size: float = 0.2, validation_size: float = 0.2
) -> Tuple[list, list, list]:
    """
    _split_names_in_training_validation_test_set Splits a list in 3 chunks representing training, validation and test set.
    Initially, there is a 80:20 split in two sets (training, validation) and (test), subsequently the (training, validation) set is split
    again 80:20 of the (training, validation) set.

    Parameters
    ----------
    names_list : list
        [description]

    Returns
    -------
    Tuple[list, list, list]
        [description]
    """
    import pandas as pd
    import numpy as np

    assert test_size > 0.0 and test_size < 1.0

    training_set_size = 1 - test_size - validation_size
    training_validation_set_size = 1 - test_size

    df = pd.DataFrame(names_list)
    training_set, validation_set, test_set = np.split(
        df.sample(frac=1),
        [int(training_set_size * len(df)), int(training_validation_set_size * len(df))],
    )

    names_training = training_set[0].tolist()
    names_validating = validation_set[0].tolist()
    names_test = test_set[0].tolist()

    assert len(names_training) + len(names_validating) + len(names_test) == len(
        names_list
    )

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
    max_epochs: int = 10,
    bulk_energy_calculation: bool = True,
    load_checkpoint: bool = True,
    names: list = [],
    load_pickled_FEC: bool = True,
    lr_AdamW: float = 1e-3,
    lr_SGD: float = 1e-3,
    weight_decay: float = 1e-2,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    include_snapshot_penalty: bool = False,
) -> Tuple[list, Number]:

    """
    Calculates the free energy of a staged free energy simulation,
    tweaks the neural net parameter so that using reweighting the difference
    between the experimental and calculated free energy is minimized.

    Parameters:
        ANImode : ANI
            The ANI model object (not instance)
        emv : str
            either 'vacuum' or droplet
        max_snapshots_per_window : int
            how many snapshots should be considered per lambda state
        checkpoint_filename : str
            filename to save/load checkpoint files. Checkpoint file is loaded only if load_checkpoint=True.
        diameter : int, opt
            diameter in Angstrom for droplet, by default = -1
        batch_size : int, opt
            by default = 1
        elements : str, opt:
            by default = 'CHON'
        names : list, opt
            for testing only! if names list is provided, no training/validation/test set split is performed but training/validation/test is performed with the names list, by default = []
        load_pickled_FEC : bool, opt
            by default = True
        lr_AdamW : float, opt
            learning rate of AdamW, by default = 1e-3
        lr_SGD: float, opt
            learning rate of SGD minimizer, by default = 1e-3
        weight_decay : float, opt
            by default = 1e-6
        test_size : float, opt
            defines the training:validation:test split
            by default = 0.2
        validation_size : float, opt
            defines the training:validation:test split
            by default = 0.2

    Raises:
        RuntimeError: [description]

    Returns:
        [type]: [description]
    """
    import datetime

    logger.info(f"Batch size: {batch_size}")

    # initialize an empty ANI model to set default parameters
    _ = ANImodel([0, 0])
    ct = datetime.datetime.now()

    # write detail of run and all parameters in the training directory
    fname = "run_info.csv"
    logger.info(f"... writing run details to {fname}  ...")
    local_variables = locals()
    with open(fname, "w+") as f:
        f.write(f"{str(ct)}\n")
        for key, value in local_variables.items():
            logger.info(f"{key}: {value}")
            f.write(f"{key}: {value}\n")

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
    pickle.dump(split, open(f"training_validation_tests.pickle", "wb+"))

    # rmsd calculation on test set
    rmse_test, dG_calc_test_initial = calculate_rmse_between_exp_and_calc(
        model=ANImodel,
        names=names_test,
        diameter=diameter,
        data_path=data_path,
        bulk_energy_calculation=bulk_energy_calculation,
        env=env,
        max_snapshots_per_window=max_snapshots_per_window,
        load_pickled_FEC=load_pickled_FEC,
        include_restraint_energy_contribution=False,
        perturbed_free_energy=False,  # NOTE: always unperturbed
    )

    print(f"RMSE on test set BEFORE optimization: {rmse_test}")

    # save batch loss through epochs
    rmse_validation = setup_and_perform_parameter_retraining(
        ANImodel=ANImodel,
        env=env,
        checkpoint_filename=checkpoint_filename,
        max_snapshots_per_window=max_snapshots_per_window,
        diameter=diameter,
        batch_size=batch_size,
        data_path=data_path,
        max_epochs=max_epochs,
        elements=elements,
        load_checkpoint=load_checkpoint,
        bulk_energy_calculation=bulk_energy_calculation,
        names_training=names_training,
        names_validating=names_validating,
        load_pickled_FEC=load_pickled_FEC,
        lr_AdamW=lr_AdamW,
        lr_SGD=lr_SGD,
        weight_decay=weight_decay,
        include_snapshot_penalty=include_snapshot_penalty,
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
        load_pickled_FEC=load_pickled_FEC,
        include_restraint_energy_contribution=False,
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
            "nn": model.optimized_neural_network.state_dict(),
            "AdamW": AdamW.state_dict(),
            "SGD": SGD.state_dict(),
            "AdamW_scheduler": AdamW_scheduler.state_dict(),
            "SGD_scheduler": SGD_scheduler.state_dict(),
        },
        latest_checkpoint,
    )


def _perform_training(
    ANImodel: ANI,
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
    load_pickled_FEC: bool,
    lr_AdamW: float,
    lr_SGD: float,
    weight_decay: float,
    include_snapshot_penalty: bool,
) -> list:

    early_stopping_learning_rate = 1.0e-8
    AdamW, AdamW_scheduler, SGD, SGD_scheduler = _get_nn_layers(
        ANImodel,
        elements=elements,
        lr_AdamW=lr_AdamW,
        lr_SGD=lr_SGD,
        weight_decay=weight_decay,
    )

    logger.info("_perform_training called ...")
    local_variables = locals()
    for key, value in local_variables.items():
        logger.info(f"{key}: {value}")

    if load_checkpoint:
        # load checkpoint file and parameters if specified
        logger.warning(f"CHECKPOINT file {checkpoint_filename} is loaded ...")
        _load_checkpoint(
            checkpoint_filename, ANImodel, AdamW, AdamW_scheduler, SGD, SGD_scheduler
        )

    base = checkpoint_filename.split(".")[0]
    best_model_checkpoint = f"{base}_best.pt"
    logger.info(f"training starting from epoch {AdamW_scheduler.last_epoch + 1}")
    logger.info(f"Writing checkpoint files to: {base}")

    # save starting point
    _save_checkpoint(
        ANImodel,
        AdamW,
        AdamW_scheduler,
        SGD,
        SGD_scheduler,
        f"{base}_{0}.pt",
    )

    ## training loop
    for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):

        # get the learning group
        learning_rate = AdamW.param_groups[0]["lr"]
        logger.debug(f"Learning rate for AdamW for current epoche: {learning_rate}")
        if learning_rate < early_stopping_learning_rate:
            print(
                "Learning rate for AdamW is lower than early_stopping_learning_reate!"
            )
            print("Stopping!")
            break

        # checkpoint -- if best parameters on validation set save parameters
        if AdamW_scheduler.is_better(rmse_validation[-1], AdamW_scheduler.best):
            torch.save(
                ANImodel.optimized_neural_network.state_dict(), best_model_checkpoint
            )

        # perform the parameter optimization and importance weighting
        _tweak_parameters(
            names_training=names_training,
            ANImodel=ANImodel,
            AdamW=AdamW,
            SGD=SGD,
            env=env,
            epoch=AdamW_scheduler.last_epoch,
            bulk_energy_calculation=bulk_energy_calculation,
            diameter=diameter,
            batch_size=batch_size,
            data_path=data_path,
            max_snapshots_per_window=max_snapshots_per_window,
            load_pickled_FEC=load_pickled_FEC,
            include_snapshot_penalty=include_snapshot_penalty,
        )

        with torch.no_grad():
            # calculate the new free energies on the validation set with optimized parameters
            current_rmse, _ = calculate_rmse_between_exp_and_calc(
                names_validating,
                model=ANImodel,
                diameter=diameter,
                data_path=data_path,
                bulk_energy_calculation=bulk_energy_calculation,
                env=env,
                max_snapshots_per_window=max_snapshots_per_window,
                perturbed_free_energy=True,
                load_pickled_FEC=load_pickled_FEC,
                include_restraint_energy_contribution=False,
            )

            rmse_validation.append(current_rmse)

        # if appropriate update LR on plateau
        AdamW_scheduler.step(rmse_validation[-1])
        SGD_scheduler.step(rmse_validation[-1])

        _save_checkpoint(
            ANImodel,
            AdamW,
            AdamW_scheduler,
            SGD,
            SGD_scheduler,
            f"{base}_{AdamW_scheduler.last_epoch}.pt",
        )

        print(
            f"RMSE on validation set: {rmse_validation[-1]} at epoch {AdamW_scheduler.last_epoch}"
        )

    return rmse_validation


def _tweak_parameters(
    names_training: list,
    ANImodel: ANI,
    AdamW,
    SGD,
    epoch: int,
    env: str,
    diameter: int,
    bulk_energy_calculation: bool,
    batch_size: int,
    data_path: str,
    max_snapshots_per_window: int,
    load_pickled_FEC: bool,
    include_snapshot_penalty: bool,
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
    SGD :
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

    logger.debug("_tweak_parameters called ...")
    # iterate over batches of molecules
    # some points to where the rational comes from:
    # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20
    it = tqdm(chunks(names_training, batch_size))
    instance_idx = 0

    # divid in batches
    for batch_idx, names in enumerate(it):
        # reset gradient
        AdamW.zero_grad()
        SGD.zero_grad()
        logger.debug(names)
        snapshot_penalty_ = []

        # divide in chunks to read in parallel
        for name_list in chunks(names, neutromeratio.constants.NUM_PROC):
            prop_list = [
                {
                    "name": name,
                    "ANImodel": ANImodel,
                    "env": env,
                    "bulk_energy_calculation": bulk_energy_calculation,
                    "data_path": data_path,
                    "max_snapshots_per_window": max_snapshots_per_window,
                    "diameter": diameter,
                    "load_pickled_FEC": load_pickled_FEC,
                    "include_restraint_energy_contribution": False,  # NOTE: beware the default value here
                }
                for name in name_list
            ]

            if neutromeratio.constants.NUM_PROC == 1:
                FEC_list = map(_setup_FEC, prop_list)

            with mp.Pool(processes=len(name_list)) as pool:

                FEC_list = pool.map(_setup_FEC, prop_list)
                pool.close()
                pool.terminate()

            # process chunks
            for fec in FEC_list:
                # count tautomer pairs
                instance_idx += 1

                loss, snapshot_penalty = _loss_function(
                    fec, fec.name, include_snapshot_penalty
                )
                snapshot_penalty_.append(snapshot_penalty)
                # gradient is calculated
                loss.backward()
                # graph is cleared here

                if include_snapshot_penalty:
                    del fec.u_ln_rho_star_wrt_parameters
            
        it.set_description(
            f"E:{epoch};B:{batch_idx+1};I:{instance_idx};SP:{torch.tensor(snapshot_penalty_).mean()} -- MSE: {loss.item()}"
        )

        # optimization steps
        AdamW.step()
        SGD.step()


def _loss_function(
    fec: FreeEnergyCalculator,
    name: str,
    include_snapshot_penalty: bool = False,
) -> Tuple[torch.Tensor, Number]:

    snapshot_penalty = torch.tensor([0.0])
    # calculate the free energies
    calc_free_energy_difference = get_perturbed_free_energy_difference(fec)
    # obtain the experimental free energies
    exp_free_energy_difference = get_experimental_values(name)
    # calculate the loss as MSE
    loss = calculate_mse(calc_free_energy_difference, exp_free_energy_difference)

    if include_snapshot_penalty:
        snapshot_penalty = fec.mae_between_potentials_for_snapshots()
        logger.debug(f"Snapshot penalty: {snapshot_penalty.item()}")
        loss += 0.5 * (snapshot_penalty ** 2)

    return loss, snapshot_penalty.item()


def _load_checkpoint(
    latest_checkpoint, model, AdamW, AdamW_scheduler, SGD, SGD_scheduler
):
    # save checkpoint
    if os.path.isfile(latest_checkpoint):
        checkpoint = torch.load(latest_checkpoint)
        model.optimized_neural_network.load_state_dict(checkpoint["nn"])
        AdamW.load_state_dict(checkpoint["AdamW"])
        SGD.load_state_dict(checkpoint["SGD"])
        AdamW_scheduler.load_state_dict(checkpoint["AdamW_scheduler"])
        SGD_scheduler.load_state_dict(checkpoint["SGD_scheduler"])
    else:
        logger.critical(f"Checkoint {latest_checkpoint} does not exist.")
        raise RuntimeError("Wanted to laod checkpoint but checkpoint does not exist")


def _get_nn_layers(
    ANImodel: ANI,
    elements: str,
    lr_AdamW: float = 1e-3,
    lr_SGD: float = 1e-3,
    weight_decay: float = 1e-6,
    layer: int = -1,
):

    """
    Extracts the trainable parameters of the defined layer for some elements of the pretrained ANI net.

    Parameters
    -------
    elements: str
    lr_AdamW: float
    lr_SGD: float
    weight_decay : float
    layer : which layer will be optimized (default -1)

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    RuntimeError
        [description]
    """

    model = ANImodel.optimized_neural_network

    if elements == "CHON":
        logger.info("Using `CHON` elements.")
        weight_layers, bias_layers = _get_nn_layers_CHON(weight_decay, layer, model)
    elif elements == "CN":
        logger.info("Using `CN` elements.")
        weight_layers, bias_layers = _get_nn_layers_CN(weight_decay, layer, model)
    elif elements == "H":
        logger.info("Using `H` elements.")
        weight_layers, bias_layers = _get_nn_layers_H(weight_decay, layer, model)
    elif elements == "C":
        logger.info("Using `C` elements.")
        weight_layers, bias_layers = _get_nn_layers_C(weight_decay, layer, model)
    else:
        raise RuntimeError(
            "Only `CHON`, `H`, `C` or `CN` as elements allowed. Aborting."
        )

    # set up minimizer for weights
    AdamW = torch.optim.AdamW(weight_layers, lr=lr_AdamW)

    # set up minimizer for bias
    SGD = torch.optim.SGD(bias_layers, lr=lr_SGD)

    AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        AdamW, "min", patience=2, verbose=True
    )  # using defailt values
    SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        SGD, "min", patience=2, verbose=True
    )  # using defailt values from https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau

    return (AdamW, AdamW_scheduler, SGD, SGD_scheduler)


def _get_nn_layers_C(weight_decay: float, layer: int, model) -> Tuple[list, list]:
    weight_layers = []
    bias_layers = []

    for nn in model:
        weight_layers.extend(
            [
                {"params": [nn.C[layer].weight], "weight_decay": weight_decay},
            ]
        )
        bias_layers.extend(
            [
                {"params": [nn.C[layer].bias]},
            ]
        )
    return (weight_layers, bias_layers)


def _get_nn_layers_H(weight_decay: float, layer: int, model) -> Tuple[list, list]:
    weight_layers = []
    bias_layers = []

    for nn in model:
        weight_layers.extend(
            [
                {"params": [nn.H[layer].weight], "weight_decay": weight_decay},
            ]
        )
        bias_layers.extend(
            [
                {"params": [nn.H[layer].bias]},
            ]
        )
    return (weight_layers, bias_layers)


def _get_nn_layers_CN(weight_decay: float, layer: int, model) -> Tuple[list, list]:
    weight_layers = []
    bias_layers = []

    for nn in model:
        weight_layers.extend(
            [
                {"params": [nn.C[layer].weight], "weight_decay": weight_decay},
                {"params": [nn.N[layer].weight], "weight_decay": weight_decay},
            ]
        )
        bias_layers.extend(
            [
                {"params": [nn.C[layer].bias]},
                {"params": [nn.N[layer].bias]},
            ]
        )
    return (weight_layers, bias_layers)


def _get_nn_layers_CHON(weight_decay: float, layer: int, model) -> Tuple[list, list]:
    weight_layers = []
    bias_layers = []

    for nn in model:
        weight_layers.extend(
            [
                {"params": [nn.C[layer].weight], "weight_decay": weight_decay},
                {"params": [nn.H[layer].weight], "weight_decay": weight_decay},
                {"params": [nn.O[layer].weight], "weight_decay": weight_decay},
                {"params": [nn.N[layer].weight], "weight_decay": weight_decay},
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
    names_training: list,
    names_validating: list,
    include_snapshot_penalty: bool,
    diameter: int = -1,
    batch_size: int = 1,
    data_path: str = "../data/",
    max_epochs: int = 10,
    elements: str = "CHON",
    load_checkpoint: bool = True,
    bulk_energy_calculation: bool = True,
    load_pickled_FEC: bool = True,
    lr_AdamW: float = 1e-3,
    lr_SGD: float = 1e-3,
    weight_decay: float = 0.000001,
) -> list:
    """
    Much of this code is taken from:
    https://aiqm.github.io/torchani/examples/nnp_training.html
    but instead of training on atomic energies the training is
    performed on relative free energies.

    The function is set up to be called from the notebook or scripts folder.

    Parameters
    -------
    ANImode : ANI
        The ANI model object (not instance)
    emv : str
        either 'vacuum' or droplet
    max_snapshots_per_window : int
        how many snapshots should be considered per lambda state
    checkpoint_filename : str
        filename to save/load checkpoint files. Checkpoint file is loaded only if load_checkpoint=True.
    diameter : int, opt
        diameter in Angstrom for droplet, by default = -1
    batch_size : int, opt
        by default = 1
    elements : str, opt:
        by default = 'CHON'
    names_training : list
        names for training set
    names_validating : list
        names for validation set
    load_pickled_FEC : bool, opt
        by default = True
    lr_AdamW : float, opt
        learning rate of AdamW, by default = 1e-3
    lr_SGD: float, opt
        learning rate of SGD minimizer, by default = 1e-3
    weight_decay : float, opt
        by default = 1e-6

    Returns
    -------
        list : rmse on validation set
    """

    assert int(batch_size) >= 1

    logger.info("setup_and_perform_parameter_retraining called ...")
    local_variables = locals()
    for key, value in local_variables.items():
        logger.info(f"{key}: {value}")

    if load_pickled_FEC:
        _ = ANImodel(
            [0, 0]
        )  # NOTE: The model needs a single initialized instance to work with pickled tautomer objects

    if env == "droplet" and diameter == -1:
        raise RuntimeError(f"Did you forget to pass the 'diamter' argument? Aborting.")

    # save batch loss through epochs
    rmse_validation = []

    # calculate the rmse on the current parameters for the validation set
    rmse_validation_set, _ = calculate_rmse_between_exp_and_calc(
        names_validating,
        model=ANImodel,
        data_path=data_path,
        bulk_energy_calculation=bulk_energy_calculation,
        env=env,
        max_snapshots_per_window=max_snapshots_per_window,
        diameter=diameter,
        perturbed_free_energy=False,
        load_pickled_FEC=load_pickled_FEC,
        include_restraint_energy_contribution=False,
    )

    rmse_validation.append(rmse_validation_set)
    print(f"RMSE on validation set: {rmse_validation[-1]} at first epoch")

    ### main training loop
    rmse_validation = _perform_training(
        ANImodel=ANImodel,
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
        load_pickled_FEC=load_pickled_FEC,
        lr_AdamW=lr_AdamW,
        lr_SGD=lr_SGD,
        weight_decay=weight_decay,
        include_snapshot_penalty=include_snapshot_penalty,
    )

    return rmse_validation


def setup_FEC(
    name: str,
    max_snapshots_per_window: int,
    ANImodel: ANI,
    bulk_energy_calculation: bool,
    env: str,
    load_pickled_FEC: bool,
    include_restraint_energy_contribution: bool,
    checkpoint_file: str = "",
    data_path: str = "../data/",
    diameter: int = -1,
    save_pickled_FEC: bool = True,  # default saves the pickled FEC
) -> FreeEnergyCalculator:

    """
    Automates the setup of the FreeEnergyCalculator object

    Parameters
    -------
    name : str
        Name of the system
    max_snapshots_per_window : int
        snapshots/lambda to use
    bulk_energy_calculation : bool
        calculate the energies in bulk
    env : bool
        environment is either `vacuum` or `droplet`
    load_pickled_FEC : bool
        if a pickled FEC is present in the location specified by data_path load it
    include_restraint_energy_contribution: bool
        include the restraint contributions in the potential energy function
    save_pickled_FEC : bool
        save the pickled FEC in the location specified by data_path, by default save_pickled_FEC = True
    Returns
    -------
    FreeEnergyCalculator

    Raises
    ------
    RuntimeError
        [description]
    RuntimeError
        [description]
    RuntimeError
        [description]
    RuntimeError
        [description]
    RuntimeError
        [description]
    """

    from neutromeratio.analysis import setup_alchemical_system_and_energy_function
    import os
    from compress_pickle import dump, load

    ANImodel([0, 0])

    def parse_lambda_from_dcd_filename(dcd_filename) -> float:
        """parse the dcd filename"""
        l = dcd_filename[: dcd_filename.find(f"_energy_in_{env}")].split("_")
        lam = l[-3]
        return float(lam)

    def _check_and_return_fec(
        fec, include_restraint_energy_contribution: bool
    ) -> FreeEnergyCalculator:
        if (
            fec.include_restraint_energy_contribution
            != include_restraint_energy_contribution
        ):
            raise RuntimeError(
                f"Attempted to load FEC with include_restraint_energy_contribution: {fec.include_restraint_energy_contribution}, but asked for include_restraint_energy_contribution: {include_restraint_energy_contribution}"
            )
        # NOTE: early exit
        return fec

    if not (env == "vacuum" or env == "droplet"):
        raise RuntimeError("Only keyword vacuum or droplet are allowed as environment.")
    if env == "droplet" and diameter == -1:
        raise RuntimeError("Something went wrong.")

    data_path = os.path.abspath(data_path)
    # check if data_path exists
    if not os.path.exists(data_path):
        raise RuntimeError(f"{data_path} does not exist!")

    fec_pickle = f"{data_path}/{name}/{name}_FEC_{max_snapshots_per_window}_for_{ANImodel.name}_restraint_{include_restraint_energy_contribution}"

    # load FEC pickle file
    if load_pickled_FEC:
        logger.debug(f"{fec_pickle}[.gz|.pickle|''] loading ...")
        if os.path.exists(f"{fec_pickle}.gz"):
            fec = load(f"{fec_pickle}.gz")
            return _check_and_return_fec(fec, include_restraint_energy_contribution)
        elif os.path.exists(f"{fec_pickle}.pickle"):
            fec = pickle.load(open(f"{fec_pickle}.pickle", "rb"))
            return _check_and_return_fec(fec, include_restraint_energy_contribution)
        elif os.path.exists(f"{fec_pickle}"):
            fec = pickle.load(open(f"{fec_pickle}", "rb"))
            return _check_and_return_fec(fec, include_restraint_energy_contribution)
        else:
            print(f"Tried to load {fec_pickle}[.gz|.pickle|''] but failed!")
            logger.critical(f"Tried to load {fec_pickle}[.gz|.pickle|''] but failed!")

    # setup alchecmial system and energy function
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

    # calculate free energy in kT
    fec = FreeEnergyCalculator(
        ani_model=energy_function,
        md_trajs=md_trajs,
        potential_energy_trajs=energies,
        lambdas=lambdas,
        bulk_energy_calculation=bulk_energy_calculation,
        max_snapshots_per_window=max_snapshots_per_window,
        include_restraint_energy_contribution=include_restraint_energy_contribution,
    )

    fec.flipped = flipped

    # save FEC
    if save_pickled_FEC:
        logger.critical(f"Saving pickled FEC to {fec_pickle}.gz")
        print(f"Saving pickled FEC to {fec_pickle}.gz")
        dump(fec, f"{fec_pickle}.gz")

    return fec


def setup_FEC_for_new_tautomer_pairs(
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
