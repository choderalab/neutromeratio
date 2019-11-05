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
from neutromeratio.constants import kT

# job idx
idx = int(sys.argv[1])
# diameter
diameter_in_angstrom = int(sys.argv[2])
# where to write the results
base_path = str(sys.argv[3])
#######################
mode = 'forward'

# read in exp results, smiles and names
exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))

exclude = ['molDWRow_1004', 'molDWRow_1110', 'molDWRow_1184', 'molDWRow_1185', 'molDWRow_1189', 'molDWRow_1262', 'molDWRow_1263',
'molDWRow_1267', 'molDWRow_1275', 'molDWRow_1279', 'molDWRow_1280', 'molDWRow_1282', 'molDWRow_1283', 'molDWRow_553',
'molDWRow_557', 'molDWRow_580', 'molDWRow_581', 'molDWRow_582', 'molDWRow_615', 'molDWRow_616', 'molDWRow_617',
'molDWRow_618', 'molDWRow_643', 'molDWRow_758', 'molDWRow_82', 'molDWRow_83', 'molDWRow_952', 'molDWRow_953',
'molDWRow_955', 'molDWRow_988', 'molDWRow_989', 'molDWRow_990', 'molDWRow_991', 'molDWRow_992']

# name of the system
protocoll = []
for name in exp_results:
    if name in exclude:
        continue
    protocoll.append(name)

name = protocoll[idx-1]
print(name)

# don't change - direction is fixed for all runs
#################
t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']


# generate both rdkit mol
tautomer = neutromeratio.Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles), nr_of_conformations=20)
if mode == 'forward':
    tautomer.perform_tautomer_transformation_forward()
elif mode == 'reverse':
    tautomer.perform_tautomer_transformation_reverse()
else:
    raise RuntimeError('No tautomer reaction direction was specified.')

tautomer.add_droplet(tautomer.hybrid_topology, 
                            tautomer.hybrid_coords, 
                            diameter=diameter_in_angstrom * unit.angstrom,
                            restrain_hydrogens=True,
                            file=f"{base_path}/{name}/{name}_in_droplet_{mode}.pdb")

# define the alchemical atoms
alchemical_atoms=[tautomer.hybrid_dummy_hydrogen, tautomer.hydrogen_idx]

print('Nr of atoms: {}'.format(len(tautomer.ligand_in_water_atoms)))


# extract hydrogen donor idx and hydrogen idx for from_mol
model = neutromeratio.ani.LinearAlchemicalDualTopologyANI(alchemical_atoms=alchemical_atoms)
model = model.to(device)
torch.set_num_threads(2)

# perform initial sampling
energy_function = neutromeratio.ANI1_force_and_energy(
                                        model = model,
                                        atoms = tautomer.ligand_in_water_atoms,
                                        mol = None,
                                        )

tautomer.add_COM_for_hybrid_ligand(np.array([diameter_in_angstrom/2, diameter_in_angstrom/2, diameter_in_angstrom/2]) * unit.angstrom)

for r in tautomer.ligand_restraints:
    energy_function.add_restraint(r)

for r in tautomer.hybrid_ligand_restraints:
    energy_function.add_restraint(r)

for r in tautomer.solvent_restraints:
    energy_function.add_restraint(r)

for r in tautomer.com_restraints:
    energy_function.add_restraint(r)












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
