import neutromeratio
from openmmtools.constants import kB
from simtk import unit
import numpy as np
import pickle
import mdtraj as md
import torchani
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import random, sys

# name of the system
name = str(sys.argv[1])
run_nr = int(sys.argv[2])
# if direction_of_tautomer_transformation == 1:
#   t1 -> t2
direction_of_tautomer_transformation = int(sys.argv[3])
perturbations_per_trial = int(sys.argv[4])
env = str(sys.argv[5])
platform = str(sys.argv[6])

exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))

if int(direction_of_tautomer_transformation) == 1:
    from_mol_tautomer_idx = 1
    to_mol_tautomer_idx = 2
else:
    from_mol_tautomer_idx = 2
    to_mol_tautomer_idx = 1

t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']

# generate both rdkit mol
mols = { 't1' : neutromeratio.generate_rdkit_mol(t1_smiles), 't2' : neutromeratio.generate_rdkit_mol(t2_smiles) }
from_mol = mols[f"t{from_mol_tautomer_idx}"]
to_mol = mols[f"t{to_mol_tautomer_idx}"]
ani_input = neutromeratio.from_mol_to_ani_input(from_mol)

if env == 'solv':
    neutromeratio.write_pdb(mols[f"t{from_mol_tautomer_idx}"], filepath=f"../data/structures/{name}/{name}_tautomer_{from_mol_tautomer_idx}_env_{env}.pdb")
    neutromeratio.add_solvent(pdb_filepath=f"../data/structures/{name}/{name}_tautomer_{from_mol_tautomer_idx}.pdb",
                            ani_input=ani_input,
                            pdb_output_filepath=f"../data/structures/{name}/{name}_tautomer_{from_mol_tautomer_idx}_env_{env}.pdb",
                            box_length = 2.5 * unit.nanometer)

    topology = md.load(f"../data/structures/{name}/{name}_tautomer_{from_mol_tautomer_idx}_env_{env}.pdb").topology
elif env == 'vac':
    neutromeratio.write_pdb(mols[f"t{from_mol_tautomer_idx}"], filepath=f"../data/structures/{name}/{name}_tautomer_{from_mol_tautomer_idx}_env_{env}.pdb")
    topology = md.load(f"../data/structures/{name}/{name}_tautomer_{from_mol_tautomer_idx}_env_{env}.pdb").topology
else:
    raise RuntimeError('Only vac and solv environments implemented.')

if platform == 'cpu':
    device = torch.device(platform)
    model = neutromeratio.ani.LinearAlchemicalANI(alchemical_atom=-1, ani_input=ani_input, device=device)
    model = model.to(device)
    torch.set_num_threads(2)
elif platform == 'cuda':
    device = torch.device(platform)
    model = neutromeratio.ani.LinearAlchemicalANI(alchemical_atom=-1, ani_input=ani_input, device=device)
    model = model.to(device)
else:
    raise RuntimeError('Only cpu and cuda environments implemented.')

# perform initial sampling

energy_function = neutromeratio.ANI1_force_and_energy(device = device,
                                          model = model,
                                          atom_list = ani_input['atom_list'],
                                          platform = platform,
                                          tautomer_transformation = {})

langevin = neutromeratio.LangevinDynamics(atom_list = ani_input['atom_list'],
                            temperature = 300*unit.kelvin,
                            force = energy_function)


n_steps = 500
x0 = np.array(ani_input['coord_list']) * unit.angstrom
equilibrium_samples, _ = langevin.run_dynamics(x0, n_steps)


# extract hydrogen donor idx and hydrogen idx for from_mol
tautomer_transformation = neutromeratio.get_tautomer_transformation(from_mol, to_mol)
model = neutromeratio.ani.LinearAlchemicalANI(alchemical_atom=tautomer_transformation['hydrogen_idx'], ani_input=ani_input, device=device, pbc=True)
model = model.to(device)

# number of time steps
work_in_runs = {}
energy_function = neutromeratio.ANI1_force_and_energy(device = device,
                                            model = model,
                                            atom_list = ani_input['atom_list'],
                                            platform = platform,
                                            tautomer_transformation = tautomer_transformation)

langevin = neutromeratio.LangevinDynamics(atom_list = ani_input['atom_list'],
                            temperature = 300*unit.kelvin,
                            force = energy_function)

hydrogen_mover = neutromeratio.NonequilibriumMC(donor_idx = tautomer_transformation['donor_idx'], 
                                        hydrogen_idx = tautomer_transformation['hydrogen_idx'], 
                                        acceptor_idx = tautomer_transformation['acceptor_idx'], 
                                        atom_list = ani_input['atom_list'], 
                                        energy_function = energy_function,
                                        langevin_dynamics= langevin)

# initial conditions: coordinates from example were given in units of angstroms   
x0 = random.choice(equilibrium_samples) 
print(f"Hydrogen {hydrogen_mover.hydrogen_idx} is moved from atom-idx {hydrogen_mover.donor_idx} to atom-idx {hydrogen_mover.acceptor_idx}.")

# run MD and MC protocoll
work, traj = hydrogen_mover.performe_md_mc_protocol(x0=x0, perturbations_per_trial=perturbations_per_trial)    
work = work['work']


# save trajectory
ani_traj = md.Trajectory(traj, topology)
ani_traj.save(f"../data/md_mc_sampling/{name}/{name}_from_t{from_mol_tautomer_idx}_to_t{to_mol_tautomer_idx}_NCMC_work_run_nr_{run_nr}_env_{env}.dcd")

# save work values 
f = open(f"../data/md_mc_sampling/{name}/{name}_from_t{from_mol_tautomer_idx}_to_t{to_mol_tautomer_idx}_NCMC_work_run_nr_{run_nr}_perturbations_per_trial_{perturbations_per_trial}_env_{env}.csv", 'w+')
for i, j in enumerate(work):
    f.write('{}, {}\n'.format(i, j))
f.close()

plt.plot(np.linspace(0, 1, len(work)), work)
plt.xlabel('lambda')
plt.ylabel('Work * kT')
plt.title('Work vs lambda value ({} steps)'.format(len(work)))
plt.savefig(f"../data/md_mc_sampling/{name}/{name}_from_t{from_mol_tautomer_idx}_to_t{to_mol_tautomer_idx}_NCMC_work_run_nr_{run_nr}_perturbations_per_trial_{perturbations_per_trial}_env_{env}.png")

print('Total work of the protocoll: {}.'.format(sum(work)))
distance_list_h_donor = []
distance_list_h_acceptor = []
for x in traj:
    distance_list_h_donor.append(neutromeratio.constants.nm_to_angstroms * np.linalg.norm(x[tautomer_transformation['hydrogen_idx']] - x[tautomer_transformation['donor_idx']]))
    distance_list_h_acceptor.append(neutromeratio.constants.nm_to_angstroms * np.linalg.norm(x[tautomer_transformation['hydrogen_idx']] - x[tautomer_transformation['acceptor_idx']]))

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(distance_list_h_donor, label='distance between donor and hydrogen', color='g')
ax1.plot(distance_list_h_acceptor, label='distance between acceptor and hydrogen', color='b')
ax2.plot(energy_function.bias_applied, label='bias applied', color='r')
ax1.set_xlabel('ts')
ax1.set_ylabel('distance [A]')
ax2.set_ylabel('energy [kJ/mol]')
plt.legend()
plt.savefig(f"../data/md_mc_sampling/{name}/{name}_from_t{from_mol_tautomer_idx}_to_t{to_mol_tautomer_idx}_NCMC_distance_run_nr_{run_nr}_perturbations_per_trial_{perturbations_per_trial}_env_{env}.png")
