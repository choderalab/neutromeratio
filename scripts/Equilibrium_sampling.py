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
# lambda state
lambda_value = float(sys.argv[2])
# platform
platform = str(sys.argv[3])
# number of steps
n_steps = int(sys.argv[4])

exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))

t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']

# generate both rdkit mol
mols = { 't1' : neutromeratio.generate_rdkit_mol(t1_smiles), 't2' : neutromeratio.generate_rdkit_mol(t2_smiles) }
from_mol = mols['t1']
to_mol = mols['t2']
ani_input = neutromeratio.from_mol_to_ani_input(from_mol, nr_of_conf=1)

tautomer_transformation = neutromeratio.get_tautomer_transformation(from_mol, to_mol)
neutromeratio.generate_hybrid_structure(ani_input, tautomer_transformation, neutromeratio.ani.ANI1_force_and_energy)

# define the alchemical atoms
alchemical_atoms=[tautomer_transformation['acceptor_hydrogen_idx'], tautomer_transformation['donor_hydrogen_idx']]

np.random.seed(0)

# extract hydrogen donor idx and hydrogen idx for from_mol
platform = 'cpu'
device = torch.device(platform)
model = neutromeratio.ani.LinearAlchemicalSingleTopologyANI(device=device, alchemical_atoms=alchemical_atoms, ani_input=ani_input)
model = model.to(device)
torch.set_num_threads(2)

# perform initial sampling
energy_function = neutromeratio.ANI1_force_and_energy(device = device,
                                          model = model,
                                          atom_list = ani_input['hybrid_atoms'],
                                          platform = platform,
                                          tautomer_transformation = tautomer_transformation)
energy_function.restrain_acceptor = True
energy_function.restrain_donor = True

langevin = neutromeratio.LangevinDynamics(atom_list = ani_input['hybrid_atoms'],
                            temperature = 300*unit.kelvin,
                            force = energy_function)

x0 = np.array(ani_input['hybrid_coords']) * unit.angstrom

energy_function.lambda_value = lambda_value
energy_function.minimize(ani_input)

equilibrium_samples, energies = langevin.run_dynamics(x0, n_steps, stepsize=1.0 * unit.femtosecond, progress_bar=True)
energies = [neutromeratio.reduced_pot(x) for x in energies]

# save equilibrium energy values 
f = open(f"../data/equilibrium_sampling/{name}/{name}_lambda_{lambda_value:0.4f}_energy.csv", 'w+')
for e in energies:
    f.write('{}\n'.format(e))
f.close()

equilibrium_samples = [x / unit.nanometer for x in equilibrium_samples]
ani_traj = md.Trajectory(equilibrium_samples, ani_input['hybrid_topology'])
ani_traj.save(f"../data/equilibrium_sampling/{name}/{name}_lambda_{lambda_value:0.4f}.dcd", force_overwrite=True)
