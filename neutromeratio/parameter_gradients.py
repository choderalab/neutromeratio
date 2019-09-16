# TODO: gradient of MBAR_estimated free energy difference w.r.t. model parameters

import numpy as np
import torch
from pymbar import MBAR
from pymbar.timeseries import detectEquilibration
from simtk import unit

from neutromeratio.ani import AlchemicalANI
from neutromeratio.constants import temperature, kT


class FreeEnergyCalculator():
    def __init__(self,
                 ani_model: AlchemicalANI,
                 ani_trajs: list,
                 potential_energy_trajs: list,
                 lambdas,
                 max_snapshots_per_window=50,
                 ):
        K = len(lambdas)
        assert (len(ani_trajs) == K)
        assert (len(potential_energy_trajs) == K)

        self.ani_model = ani_model
        self.ani_trajs = ani_trajs
        self.potential_energy_trajs = potential_energy_trajs
        self.lambdas = lambdas

        # thin each based automatic equilibration detection
        N_k = []

        snapshots = []
        for i in range(K):
            traj = self.ani_trajs[i]

            equil, g = detectEquilibration(self.potential_energy_trajs[i])[:2]
            thinning = int(g)
            if len(traj[equil::thinning]) > max_snapshots_per_window:
                # what thinning will give me len(traj[equil::thinning]) == max_snapshots_per_window?
                thinning = int((len(traj) - equil) / max_snapshots_per_window)

            new_snapshots = list(traj[equil::thinning].xyz * unit.nanometer)[:max_snapshots_per_window]
            N_k.append(len(new_snapshots))
            snapshots.extend(new_snapshots)

        self.snapshots = snapshots
        N = len(snapshots)
        u_kn = np.zeros((K, N))
        for k in range(K):
            lamb = lambdas[k]
            for n in range(N):
                u_kn[k, n] = self.ani_model.calculate_energy(snapshots[n], lamb) / kT
        self.mbar = MBAR(u_kn, N_k)

    def compute_perturbed_free_energies(self, u_ln):
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
        u_0 = torch.tensor(
            [self.ani_model.calculate_energy(s, lambda_value = 0) / kT for s in self.snapshots],
            dtype=torch.double, requires_grad=True,
        )
        # TODO: vectorize!
        u_1 = torch.tensor(
            [self.ani_model.calculate_energy(s, lambda_value = 1) / kT for s in self.snapshots],
            dtype=torch.double, requires_grad=True,
        )
        u_ln = torch.stack([u_0, u_1])
        return u_ln

    def compute_free_energy_difference(self):
        u_ln = self.form_u_ln()
        f_k = self.compute_perturbed_free_energies(u_ln)
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
                                              temperature=temperature,
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
