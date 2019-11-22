# TODO: gradient of MBAR_estimated free energy difference w.r.t. model parameters

import numpy as np
import torch
from pymbar import MBAR
import logging
from pymbar.timeseries import detectEquilibration
from simtk import unit
from tqdm import tqdm

from neutromeratio.ani import AlchemicalANI
from neutromeratio.constants import kT, hartree_to_kJ_mol

logger = logging.getLogger(__name__)


class FreeEnergyCalculator():
    def __init__(self,
                 ani_model: AlchemicalANI,
                 ani_trajs: list,
                 potential_energy_trajs: list,
                 lambdas,
                 n_atoms:int,
                 per_atom_stddev_treshold:float=0.5,
                 max_snapshots_per_window=50,
                 ):
        
        K = len(lambdas)
        assert (len(ani_trajs) == K)
        assert (len(potential_energy_trajs) == K)
        logging.info(f"Per atom treshold used for filtering: {per_atom_stddev_treshold}")
        self.ani_model = ani_model
        self.potential_energy_trajs = potential_energy_trajs # for detecting equilibrium
        self.lambdas = lambdas
        self.ani_trajs = ani_trajs
        self.n_atoms = n_atoms

        N_k, snapshots, used_lambdas = self.remove_confs_with_high_stddev(max_snapshots_per_window, per_atom_stddev_treshold)

        # end-point energies, bias, stddev
        lambda0_e_b_stddev = [self.ani_model.calculate_energy(x, lambda_value=0.0) for x in tqdm(snapshots)]
        lambda1_e_b_stddev = [self.ani_model.calculate_energy(x, lambda_value=1.0) for x in tqdm(snapshots)]

        # extract endpoint energies
        lambda0_e = [e/kT for e in [e_b_stddev[0] for e_b_stddev in lambda0_e_b_stddev]]
        lambda1_e = [e/kT for e in [e_b_stddev[0] for e_b_stddev in lambda1_e_b_stddev]]

        def get_mix(lambda0, lambda1, lam=0.0):
            return (1 - lam) * np.array(lambda0) + lam * np.array(lambda1)

        print('Nr of atoms: {}'.format(n_atoms))
        
        u_kn = np.stack(
            [get_mix(lambda0_e, lambda1_e, lam) for lam in sorted(used_lambdas)]
            )
        
        self.mbar = MBAR(u_kn, N_k)

    def remove_confs_with_high_stddev(self, max_snapshots_per_window:int, per_atom_thresh):
        
        def calculate_stddev(snapshots):
            lambda0_e_b_stddev = [self.ani_model.calculate_energy(x, lambda_value=0.0) for x in tqdm(snapshots)]
            lambda1_e_b_stddev = [self.ani_model.calculate_energy(x, lambda_value=1.0) for x in tqdm(snapshots)]

            # extract endpoint stddev
            lambda0_stddev = [stddev/kT for stddev in [e_b_stddev[2] for e_b_stddev in lambda0_e_b_stddev]]
            lambda1_stddev = [stddev/kT for stddev in [e_b_stddev[2] for e_b_stddev in lambda1_e_b_stddev]]
            return np.array(lambda0_stddev), np.array(lambda1_stddev)

        def compute_linear_penalty(current_stddev):
            total_thresh = (per_atom_thresh * self.n_atoms) * unit.kilojoule_per_mole
            linear_penalty = np.maximum(0, current_stddev - (total_thresh/kT))
            return linear_penalty

        def compute_last_valid_ind(linear_penalty):
            return np.argmax(np.cumsum(linear_penalty) > 0) -1
            
        ani_trajs = {}
        for lam, traj, potential_energy in zip(self.lambdas, self.ani_trajs, self.potential_energy_trajs):
            equil, g = detectEquilibration(potential_energy)[:2]
            snapshots = list(traj[equil:].xyz * unit.nanometer)[:max_snapshots_per_window] 
            if len(snapshots) == 0: # otherwise we will get problems down the line
                middle_of_traj = len(traj)/2
                logger.warning(f"No equilibrium length detected in snapshots for lambda: {lam}")
                logger.warning(f"Taking 50 snapshots from middle of simulation.")
                snapshots = list(traj[middle_of_traj:middle_of_traj+50].xyz * unit.nanometer)
            ani_trajs[lam] = snapshots

        last_valid_inds = {}
        logging.info(f"Per atom treshold used for filtering: {per_atom_thresh}")
        logging.info(f"Max snapshots per lambda: {max_snapshots_per_window}")
        
        for lam in ani_trajs:
            lambda0_stddev, lambda1_stddev = calculate_stddev(ani_trajs[lam])
            current_stddev = (1 - lam) * lambda0_stddev + lam * lambda1_stddev
            if per_atom_thresh < 0.0:
                last_valid_ind = -1
            else:
                linear_penalty = compute_linear_penalty(current_stddev)
                last_valid_ind = compute_last_valid_ind(linear_penalty)
            last_valid_inds[lam] = last_valid_ind

        lambdas_with_usable_samples = []
        for lam in sorted(list(last_valid_inds.keys())):
            if last_valid_inds[lam] > 5 or last_valid_inds[lam] == -1: # -1 means all can be used
                lambdas_with_usable_samples.append(lam)

        snapshots = []
        N_k = []
        max_n_snapshots_per_state = 10

        for lam in lambdas_with_usable_samples:
            traj = ani_trajs[lam][0:last_valid_inds[lam]]
            further_thinning = 1
            if len(traj) > max_n_snapshots_per_state:
                further_thinning = int(len(traj) / max_n_snapshots_per_state)
            new_snapshots = traj[::further_thinning]
            snapshots.extend(new_snapshots)
            N_k.append(len(new_snapshots))

        return N_k, snapshots, lambdas_with_usable_samples


    @property
    def free_energy_differences(self):
        """matrix of free energy differences"""
        return self.mbar.getFreeEnergyDifferences()[0]
    
    @property
    def free_energy_difference_uncertainties(self):
        """matrix of asymptotic uncertainty-estimates accompanying free energy differences"""
        return self.mbar.getFreeEnergyDifferences()[1]
    
    @property
    def end_state_free_energy_difference(self):
        """DeltaF[lambda=1 --> lambda=0]"""
        DeltaF_ij, dDeltaF_ij, _ = self.mbar.getFreeEnergyDifferences()
        K = len(DeltaF_ij)
        return DeltaF_ij[0, K-1], dDeltaF_ij[0, K-1]



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
        e0_e_b_stddev = [self.ani_model.calculate_energy(s, lambda_value = 0) for s in self.snapshots]
        u0_stddev = [e_b_stddev[2] / kT for e_b_stddev in e0_e_b_stddev] 
        u_0 = torch.tensor(
            [e_b_stddev[0] / kT for e_b_stddev in e0_e_b_stddev],
            dtype=torch.double, requires_grad=True,
        )
        # TODO: vectorize!
        e1_e_b_stddev = [self.ani_model.calculate_energy(s, lambda_value = 1) for s in self.snapshots]
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
    neutromeratio.generate_hybrid_structure(ani_input, tautomer_transformation, neutromeratio.ani.ANI1_force_and_energy)
    alchemical_atoms = [tautomer_transformation['acceptor_hydrogen_idx'], tautomer_transformation['donor_hydrogen_idx']]

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
