# This script runs staged free energy calculations 
# given a system name, a lambda value and the number of equilibrium 
# steps

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
from neutromeratio.constants import platform, device

# name of the system
name = str(sys.argv[1])
# lambda state
lambda_value = float(sys.argv[2])
# number of steps
n_steps = int(sys.argv[3])

exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))

t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']

tautomer = neutromeratio.Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles), nr_of_conformations=1)

neutromeratio.generate_hybrid_structure(ani_input, tautomer_transformation)

# define the alchemical atoms
alchemical_atoms=[tautomer_transformation['acceptor_hydrogen_idx'], tautomer_transformation['donor_hydrogen_idx']]

np.random.seed(0)

# extract hydrogen donor idx and hydrogen idx for from_mol
model = neutromeratio.ani.LinearAlchemicalSingleTopologyANI(alchemical_atoms=alchemical_atoms, ani_input=ani_input)
model = model.to(device)
torch.set_num_threads(2)

# perform initial sampling
energy_function = neutromeratio.ANI1_force_and_energy(
                                          model = model,
                                          atoms = ani_input['hybrid_atoms'],
                                          )


langevin = neutromeratio.LangevinDynamics(atoms = ani_input['hybrid_atoms'],
                            temperature = 300*unit.kelvin,
                            force = energy_function)

x0 = np.array(ani_input['hybrid_coords']) * unit.angstrom

# add constraints to energy function!
for e in ani_input['hybrid_restraints']:
    energy_function.add_restraint(e)

# minimize
energy_function.minimize(ani_input)

# run simulation
equilibrium_samples, energies = langevin.run_dynamics(x0, lambda_value=lambda_value, n_steps=n_steps, stepsize=1.0 * unit.femtosecond, progress_bar=True)
energies = [neutromeratio.reduced_pot(x) for x in energies]

# save equilibrium energy values 
f = open(f"../data/equilibrium_sampling/{name}/{name}_lambda_{lambda_value:0.4f}_energy.csv", 'w+')
for e in energies:
    f.write('{}\n'.format(e))
f.close()

equilibrium_samples = [x.value_in_unit(unit.nanometer) for x in equilibrium_samples]
ani_traj = md.Trajectory(equilibrium_samples, ani_input['hybrid_topology'])
ani_traj.save(f"../data/equilibrium_sampling/{name}/{name}_lambda_{lambda_value:0.4f}.dcd", force_overwrite=True)
