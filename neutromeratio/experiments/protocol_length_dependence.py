# loop over protocols of length [2,4,8,16,32,64,128]

import pickle

import numpy as np
import torch
from simtk import unit

import neutromeratio

with open('../../data/exp_results.pickle', 'rb') as f:
    exp_results = pickle.load(f)


def build_hydrogen_mover(name='molDWRow_298'):
    from_mol_tautomer_idx = 1
    to_mol_tautomer_idx = 2

    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    # generate both rdkit mol
    mols = {'t1': neutromeratio.generate_rdkit_mol(t1_smiles), 't2': neutromeratio.generate_rdkit_mol(t2_smiles)}
    from_mol = mols[f"t{from_mol_tautomer_idx}"]
    to_mol = mols[f"t{to_mol_tautomer_idx}"]
    ani_input = neutromeratio.from_mol_to_ani_input(from_mol)
    tautomer_transformation = neutromeratio.get_donor_atom_idx(from_mol, to_mol)

    atom_list = ani_input['atom_list']
    hydrogen_idx = tautomer_transformation['hydrogen_idx']
    donor_idx = tautomer_transformation['donor_idx']
    acceptor_idx = tautomer_transformation['acceptor_idx']
    n_atoms = len(atom_list)

    x0 = ani_input['coord_list']
    platform = 'cpu'
    device = torch.device(platform)
    model = neutromeratio.ani.LinearAlchemicalANI(alchemical_atoms=hydrogen_idx)
    model = model.to(device)
    energy_function = neutromeratio.ANI1_force_and_energy(device=device,
                                                          model=model,
                                                          atom_list=atom_list,
                                                          platform=platform,
                                                          tautomer_transformation=tautomer_transformation)
    energy_function.calculate_energy(x0)

    from neutromeratio.constants import temperature
    langevin = neutromeratio.LangevinDynamics(atom_list=atom_list,
                                              temperature=temperature,
                                              force=energy_function)

    hydrogen_mover = neutromeratio.NonequilibriumMC(donor_idx=donor_idx,
                                                    hydrogen_idx=hydrogen_idx,
                                                    acceptor_idx=acceptor_idx,
                                                    atom_list=atom_list,
                                                    energy_function=energy_function,
                                                    langevin_dynamics=langevin)
    return hydrogen_mover


def load_md_samples(name='molDWRow_298'):
    import mdtraj as md
    # TODO: this hard-codes t1, but in other parts of this script we look for a `from_mol_tautomer_idx`
    path_to_t1_top = '../../data/md_sampling/{name}/{name}_tautomer_1.pdb'.format(name=name)
    path_to_t1_structures = '../../data/md_sampling/{name}/{name}_t1_run1_anicxx.dcd'.format(name=name)

    t1_traj = md.load_dcd(path_to_t1_structures, path_to_t1_top)
    return t1_traj.xyz


if __name__ == '__main__':
    name = 'molDWRow_298'
    hydrogen_mover = build_hydrogen_mover(name)
    samples = load_md_samples(name)


    def draw_t1_sample():
        return samples[np.random.randint(len(samples))] * unit.angstrom


    def simulate_with_protocol_length(protocol_length=2, n_samples=50):
        results = []
        for _ in range(n_samples):
            x0 = draw_t1_sample()
            result = hydrogen_mover.perform_md_mc_protocol(x0, nr_of_mc_trials=protocol_length)
            results.append(result)
        with open('results_length={protocol_length}.pkl'.format(protocol_length=protocol_length), 'wb') as f:
            pickle.dump(results, f)
        works = np.array([sum(w) for (w, traj) in results])
        return works


    import matplotlib.pyplot as plt

    protocol_lengths = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    n_samples = 5
    work_stddevs = []
    for protocol_length in protocol_lengths:
        works = simulate_with_protocol_length(protocol_length, n_samples=n_samples)
        print('mean(works) at protocol_length={}:   '.format(protocol_length) + '{:.3f} kT'.format(np.mean(works)))
        print('stddev(works) at protocol_length={}: '.format(protocol_length) + '{:.3f} kT'.format(np.std(works)))
        work_stddevs.append(np.std(works))

        if len(work_stddevs) > 1:
            ax = plt.subplot(1, 1, 1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.plot(protocol_lengths[:len(work_stddevs)], work_stddevs)
            plt.xlabel('protocol length (fs)')  # TODO: don't hardcode assumption of timestep, etc.
            plt.ylabel('work standard deviation (kT)')
            plt.xscale('log')
            plt.yscale('log')
            plt.title('protocol-length dependence\n(stddev estimated from {} samples)'.format(n_samples))
            plt.savefig('protocol_length_dependence.png', dpi=300, bbox_inches='tight')
            plt.close()
