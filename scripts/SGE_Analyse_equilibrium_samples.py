import neutromeratio
from openmmtools.constants import kB
from simtk import unit
import numpy as np
import pickle
import mdtraj as md
import matplotlib.pyplot as plt
import sys
import torch
from neutromeratio.parameter_gradients import FreeEnergyCalculator
from neutromeratio.constants import temperature
#######################
kT = kB * temperature

# read in exp results, smiles and names
exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))

# name of the system
name = str(sys.argv[1])

# don't change - direction is fixed for all runs
#################
t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']

tautomer = neutromeratio.Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles), nr_of_conformations=1)

# get tautomer transformation
tautomer.perform_tautomer_transformation_forward()
# define the alchemical atoms
alchemical_atoms=[tautomer_transformation['acceptor_hydrogen_idx'], tautomer_transformation['donor_hydrogen_idx']]

# generate the energy function
platform = 'cpu'
device = torch.device(platform)


model = neutromeratio.ani.LinearAlchemicalSingleTopologyANI(alchemical_atoms=alchemical_atoms, ani_input=ani_input)
model = model.to(device)
torch.set_num_threads(2)

# perform initial sampling
energy_function = neutromeratio.ANI1_force_and_energy(
                                          model = model,
                                          atoms = ani_input['hybrid_atoms']
                                          )
# add constraints to energy function!
for e in ani_input['hybrid_restraints']:
    energy_function.add_restraint(e)

# 20 steps inclusive endpoints
# getting all the energies, snapshots and lambda values in lists
# NOTE: This will be changed in the future
lambda_value = 0.0
energies = []
ani_trajs = []
lambdas = []
for _ in range(20):
    lambdas.append(lambda_value)
    f_traj = f"/home/mwieder/Work/Projects/neutromeratio/data/equilibrium_sampling/{name}/{name}_lambda_{lambda_value:0.4f}.dcd"
    traj = md.load_dcd(f_traj, top=ani_input['hybrid_topology'])
    ani_trajs.append(traj)
    
    f = open(f"/home/mwieder/Work/Projects/neutromeratio/data/equilibrium_sampling/{name}/{name}_lambda_{lambda_value:0.4f}_energy.csv", 'r')  
    tmp_e = []
    for e in f:
        tmp_e.append(float(e))
    f.close()
    
    energies.append(np.array(tmp_e))
    lambda_value +=0.05

# plotting the energies of all equilibrium runs
for e in energies: 
    plt.plot(e, alpha=0.5)
plt.show()
plt.savefig(f"/home/mwieder/Work/Projects/neutromeratio/data/equilibrium_sampling/{name}/{name}_energy.png")

# calculate free energy in kT
fec = FreeEnergyCalculator(ani_model=energy_function, ani_trajs=ani_trajs, potential_energy_trajs=energies, lambdas=lambdas)
free_energy_in_kT = fec.compute_free_energy_difference()
f = open('/home/mwieder/Work/Projects/neutromeratio/data/equilibrium_sampling/energies.csv', 'a+')
f.write(f"{name}, {free_energy_in_kT}\n")
f.close()
