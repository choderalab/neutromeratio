# TODO: gradient of MBAR_estimated free energy difference w.r.t. model parameters

import logging

import numpy as np
import torch
from pymbar import MBAR
from pymbar.timeseries import detectEquilibration
from simtk import unit
from tqdm import tqdm

from neutromeratio.ani import AlchemicalANI
from neutromeratio.constants import hartree_to_kJ_mol, kT

logger = logging.getLogger(__name__)


class FreeEnergyCalculator():
    def __init__(self,
                 ani_model: AlchemicalANI,
                 ani_trajs: list,
                 potential_energy_trajs: list,
                 lambdas: list,
                 n_atoms: int,
                 per_atom_thresh: unit.Quantity = 0.5*unit.kilojoule_per_mole,
                 max_snapshots_per_window=50,
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
        n_atoms : int
            number of atoms
        per_atom_thresh : float
            exclude snapshots where ensemble stddev in energy / n_atoms exceeds this threshold, in kJ/mol
            if per_atom_tresh == -1 ignores ensemble stddev

        """
        K = len(lambdas)
        assert (len(ani_trajs) == K)
        assert (len(potential_energy_trajs) == K)
        logging.info(f"Per atom threshold used for filtering: {per_atom_thresh}")
        self.ani_model = ani_model
        self.potential_energy_trajs = potential_energy_trajs  # for detecting equilibrium
        self.lambdas = lambdas
        self.ani_trajs = ani_trajs
        self.n_atoms = n_atoms

        N_k, snapshots, used_lambdas = self.remove_confs_with_high_stddev(max_snapshots_per_window, per_atom_thresh)

        # end-point energies, restraint_bias, stddev
        lambda0_e_b_stddev = [self.ani_model.calculate_energy(x, lambda_value=0.0) for x in tqdm(snapshots)]
        lambda1_e_b_stddev = [self.ani_model.calculate_energy(x, lambda_value=1.0) for x in tqdm(snapshots)]

        # extract endpoint energies
        lambda0_e = [e/kT for e in [e_b_stddev[0] for e_b_stddev in lambda0_e_b_stddev]]
        lambda1_e = [e/kT for e in [e_b_stddev[0] for e_b_stddev in lambda1_e_b_stddev]]

        def get_mix(lambda0, lambda1, lam=0.0):
            return (1 - lam) * np.array(lambda0) + lam * np.array(lambda1)

        logger.info('Nr of atoms: {}'.format(n_atoms))

        u_kn = np.stack(
            [get_mix(lambda0_e, lambda1_e, lam) for lam in sorted(used_lambdas)]
        )

        self.mbar = MBAR(u_kn, N_k)

    def remove_confs_with_high_stddev(self, max_snapshots_per_window: int, per_atom_thresh: float):
        """
        Removes conformations with ensemble energy stddev per atom above a given threshold.
        Parameters
        ----------
        max_snapshots_per_window : int
            maximum number of conformations per lambda window
        per_atom_thresh : float
            per atom stddev threshold - by default this is set to 0.5 kJ/mol/atom
        """

        def calculate_stddev(snapshots):
            # calculate energy, restraint_bias and stddev for endstates
            logger.info('Calculating stddev')
            lambda0_e_b_stddev = [self.ani_model.calculate_energy(x, lambda_value=0.0) for x in tqdm(snapshots)]
            lambda1_e_b_stddev = [self.ani_model.calculate_energy(x, lambda_value=1.0) for x in tqdm(snapshots)]

            # extract endpoint stddev and return
            lambda0_stddev = [stddev/kT for stddev in [e_b_stddev[2] for e_b_stddev in lambda0_e_b_stddev]]
            lambda1_stddev = [stddev/kT for stddev in [e_b_stddev[2] for e_b_stddev in lambda1_e_b_stddev]]
            return np.array(lambda0_stddev), np.array(lambda1_stddev)

        def compute_linear_ensemble_bias(current_stddev):
            # calculate the total energy stddev threshold based on the provided per_atom_thresh
            # and the number of atoms
            total_thresh = (per_atom_thresh * self.n_atoms)
            logger.info(f"Per system treshold: {total_thresh}")
            # if stddev for a given conformation < total_thresh => 0.0
            # if stddev for a given conformation > total_thresh => stddev - total_threshold
            linear_ensemble_bias = np.maximum(0, current_stddev - (total_thresh/kT))
            return linear_ensemble_bias

        def compute_last_valid_ind(linear_ensemble_bias):
            # return the idx of the first entry that is above 0.0
            if np.sum(linear_ensemble_bias) < 0.01:  # means all can be used
                logger.info('Last valid ind: -1')
                return -1
            elif np.argmax(np.cumsum(linear_ensemble_bias) > 0) == 0:  # means nothing can be used
                logger.info('Last valid ind: 0')
                return 0
            else:
                idx = np.argmax(np.cumsum(linear_ensemble_bias) > 0) - 1
                logger.info(f"Last valid ind: {idx}")
                return idx  # means up to idx can be used

        ani_trajs = {}
        further_thinning = 10
        for lam, traj, potential_energy in zip(self.lambdas, self.ani_trajs, self.potential_energy_trajs):
            # detect equilibrium
            #equil, g = detectEquilibration(potential_energy)[:2]
            # thinn snapshots and return max_snapshots_per_window confs
            #snapshots = list(traj[int(len(traj)/2):].xyz * unit.nanometer)[::further_thinning]
            start = int(len(traj) * 0.2) # remove first 20%
            snapshots = list(traj[start:].xyz * unit.nanometer)[:max_snapshots_per_window]
            ani_trajs[lam] = snapshots
            logger.info(f"Snapshots per lambda: {len(snapshots)}")

        last_valid_inds = {}
        logger.info(f"Looking through {len(ani_trajs)} lambda windows")
        for lam in sorted(ani_trajs.keys()):
            if per_atom_thresh/kT < 0:
                last_valid_ind = -1
            else:
                logger.info(f"Calculating stddev and penarly for lambda: {lam}")
                # calculate endstate stddev for given confs
                lambda0_stddev, lambda1_stddev = calculate_stddev(ani_trajs[lam])
                # scale for current lam
                current_stddev = (1 - lam) * lambda0_stddev + lam * lambda1_stddev
                linear_ensemble_bias = compute_linear_ensemble_bias(current_stddev)
                last_valid_ind = compute_last_valid_ind(linear_ensemble_bias)
            last_valid_inds[lam] = last_valid_ind

        snapshots = []
        self.snapshot_mask = []
        N_k = []
        lambdas_with_usable_samples = []
        # lambbdas with usable samples applied to usable conformations
        for lam in sorted(list(last_valid_inds.keys())):
            logger.info(f"lam: {lam}")
            if last_valid_inds[lam] > 5:
                lambdas_with_usable_samples.append(lam)
                new_snapshots = ani_trajs[lam][0:last_valid_inds[lam]]
                logger.info(
                    f"For lambda {lam}: {len(new_snapshots)} snaphosts below treshold out of {len(ani_trajs[lam])}")
                snapshots.extend(new_snapshots)
                N_k.append(len(new_snapshots))
                self.snapshot_mask.append(len(new_snapshots))
            elif last_valid_inds[lam] == -1:  # -1 means all can be used
                lambdas_with_usable_samples.append(lam)
                new_snapshots = ani_trajs[lam][:]
                logger.info(
                    f"For lambda {lam}: {len(new_snapshots)} snaphosts below treshold out of {len(ani_trajs[lam])}")
                snapshots.extend(new_snapshots)
                N_k.append(len(new_snapshots))
                self.snapshot_mask.append(len(new_snapshots))

            else:
                self.snapshot_mask.append(0)
                logger.info(f"For lambda {lam}: ZERO snaphosts below treshold out of {len(ani_trajs[lam])}")
        logger.info(f"Nr of snapshots considered for postprocessing: {len(snapshots)}")
        return N_k, snapshots, lambdas_with_usable_samples

    @property
    def free_energy_differences(self):
        """matrix of free energy differences"""
        return self.mbar.getFreeEnergyDifferences(return_dict=True)['Delta_f']

    @property
    def free_energy_difference_uncertainties(self):
        """matrix of asymptotic uncertainty-estimates accompanying free energy differences"""
        return self.mbar.getFreeEnergyDifferences(return_dict=True)['dDelta_f']

    @property
    def end_state_free_energy_difference(self):
        """DeltaF[lambda=1 --> lambda=0]"""
        results = self.mbar.getFreeEnergyDifferences(return_dict=True)
        return results['Delta_f'][0, -1], results['dDelta_f'][0, -1]

    def compute_perturbed_free_energies(self, u_ln, u0_stddev, u1_stddev):
        """compute perturbed free energies at new thermodynamic states l"""
        assert (type(u_ln) == torch.Tensor)

        def torchify(x):
            return torch.tensor(x, dtype=torch.double, requires_grad=True)

        states_with_samples = torch.tensor(self.mbar.N_k > 0)
        N_k = torch.tensor(self.mbar.N_k, dtype=torch.double)
        f_k = torchify(self.mbar.f_k)
        u_kn = torchify(self.mbar.u_kn)

        log_q_k = f_k[states_with_samples] - u_kn[states_with_samples].T
        # TODO: double check that torch.logsumexp(x + torch.log(b)) is the same as scipy.special.logsumexp(x, b=b)
        A = log_q_k + torch.log(N_k[states_with_samples])
        log_denominator_n = torch.logsumexp(A, dim=1)

        B = - u_ln - log_denominator_n
        return - torch.logsumexp(B, dim=1)

    def form_u_ln(self):

        # TODO: vectorize!
        e0_e_b_stddev = [self.ani_model.calculate_energy(s, lambda_value=0) for s in self.snapshots]
        u0_stddev = [e_b_stddev[2] / kT for e_b_stddev in e0_e_b_stddev]
        u_0 = torch.tensor(
            [e_b_stddev[0] / kT for e_b_stddev in e0_e_b_stddev],
            dtype=torch.double, requires_grad=True,
        )
        # TODO: vectorize!
        e1_e_b_stddev = [self.ani_model.calculate_energy(s, lambda_value=1) for s in self.snapshots]
        u1_stddev = [e_b_stddev[2] / kT for e_b_stddev in e1_e_b_stddev]
        u_1 = torch.tensor(
            [e_b_stddev[0] / kT for e_b_stddev in e1_e_b_stddev],
            dtype=torch.double, requires_grad=True,
        )
        u_ln = torch.stack([u_0, u_1])
        return u_ln,  u0_stddev, u1_stddev

    def compute_free_energy_difference(self):
        u_ln, u0_stddev, u1_stddev = self.form_u_ln()
        f_k = self.compute_perturbed_free_energies(u_ln, u0_stddev, u1_stddev)
        return f_k[1] - f_k[0]








class FreeEnergyCalculatorKappa(FreeEnergyCalculator):
        
    def __init__(self,
                ani_model: AlchemicalANI,
                ani_trajs: list,
                kappas: list,
                n_atoms: int,
                added_restraint : list,
                max_snapshots_per_window=50,
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
        kappas : list
            all lambda states
        n_atoms : int
            number of atoms
        per_atom_thresh : float
            exclude snapshots where ensemble stddev in energy / n_atoms exceeds this threshold, in kJ/mol
            if per_atom_tresh == -1 ignores ensemble stddev

        """
        K = len(kappas)
        assert (len(ani_trajs) == K)
        assert (len(potential_energy_trajs) == K)
        
        self.ani_model = ani_model
        self.kappas = kappas
        self.ani_trajs = ani_trajs
        self.n_atoms = n_atoms

        N_k = []
        snapshots = []
        tmp = {}

        for kappa, traj in zip(kappas, ani_trajs):
            start = int(len(traj) * 0.2) # remove first 20%
            tmp[kappa] = traj[start:-1].xyz * unit.nanometer
        for kappa in sorted(kappas):
            N_k.append(len(tmp[kappa]))
            snapshots.extend(tmp[kappa])

        e_list = [ani_model.calculate_energy(x, lambda_value=1.0)[0] / kT for x in tqdm(snapshots)]
        u_kn = np.stack([e_list for _ in range(len(N_k))])
        bias = np.stack(
            [[(bias.restraint(x).detach().numpy()) * unit.kilojoule_per_mole / kT for x in snapshots] for bias in added_restraint]
        )
        self.mbar = MBAR(u_kn + bias, N_k)




if __name__ == '__main__':
    import neutromeratio
    import pickle
    import mdtraj as md
    from tqdm import tqdm

    exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))

    # specify the system you want to simulate
    name = 'molDWRow_298'
    # name = 'molDWRow_37'
    # name = 'molDWRow_45'
    # name = 'molDWRow_160'
    # name = 'molDWRow_590'

    from_mol_tautomer_idx = 1
    to_mol_tautomer_idx = 2

    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    # generate both rdkit molhttp://localhost:8888/notebooks/notebooks/testing-hybrid-structures.ipynb#
    mols = {'t1': neutromeratio.generate_rdkit_mol(t1_smiles), 't2': neutromeratio.generate_rdkit_mol(t2_smiles)}
    from_mol = mols[f"t{from_mol_tautomer_idx}"]
    to_mol = mols[f"t{to_mol_tautomer_idx}"]
    ani_input = neutromeratio.from_mol_to_ani_input(from_mol)

    tautomer_transformation = neutromeratio.get_tautomer_transformation(from_mol, to_mol)
    neutromeratio.generate_hybrid_structure(ani_input, tautomer_transformation,
                                            neutromeratio.ani.ANI1_force_and_energy)
    alchemical_atoms = [tautomer_transformation['acceptor_hydrogen_idx'],
                        tautomer_transformation['donor_hydrogen_idx']]

    # extract hydrogen donor idx and hydrogen idx for from_mol
    platform = 'cpu'
    device = torch.device(platform)
    model = neutromeratio.ani.LinearAlchemicalSingleTopologyANI(device=device, alchemical_atoms=alchemical_atoms,
                                                                ani_input=ani_input)
    model = model.to(device)

    # perform initial sampling
    ani_trajs = []
    n_steps = 100
    ani_model = neutromeratio.ANI1_force_and_energy(device=device,
                                                    model=model,
                                                    atom_list=ani_input['hybrid_atoms'],
                                                    platform=platform,
                                                    tautomer_transformation=tautomer_transformation)
    ani_model.restrain_acceptor = True
    ani_model.restrain_donor = True

    langevin = neutromeratio.LangevinDynamics(atom_list=ani_input['hybrid_atoms'],
                                              force=ani_model)

    x0 = np.array(ani_input['hybrid_coords']) * unit.angstrom
    potential_energy_trajs = []

    from tqdm import tqdm

    lambdas = np.linspace(0, 1, 5)
    for lamb in tqdm(lambdas):
        ani_model.lambda_value = lamb

        equilibrium_samples, energies = langevin.run_dynamics(x0, n_steps)
        potential_energy_trajs.append(np.array(
            [e.value_in_unit(unit.kilojoule_per_mole) for e in energies]
        ))
        equilibrium_samples = [x / unit.nanometer for x in equilibrium_samples]
        ani_traj = md.Trajectory(equilibrium_samples, ani_input['hybrid_topolog'])

        ani_trajs.append(ani_traj)

    free_energy_calculator = FreeEnergyCalculator(
        ani_model, ani_trajs, potential_energy_trajs, lambdas,
    )

    deltaF = free_energy_calculator.compute_free_energy_difference()
    print(deltaF)

    # let's say I had a loss function that wanted the free energy difference
    # estimate to be equal to 6:
    L = (deltaF - 6) ** 2

    # can I backpropagate derivatives painlessly to the ANI neural net parameters?
    L.backward()  # no errors or warnings

    params = list(ani_model.model.parameters())
    for p in params:
        print(p.grad)  # all None
