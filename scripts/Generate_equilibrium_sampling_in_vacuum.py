from simtk import unit
import numpy as np
from tqdm import tqdm
import mdtraj as md
import nglview
from rdkit import Chem
from rdkit.Chem import AllChem
import neutromeratio
import matplotlib.pyplot as plt
import pickle
import torchani
import torch
from neutromeratio.constants import device, platform, kT, exclude_set
import sys, os


# name of the system
idx = int(sys.argv[1])
# number of steps
n_steps = int(sys.argv[2])
# where to write the results
base_path = str(sys.argv[3])

mode = 'forward'

protocol = []
exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))
for name in sorted(exp_results):
    if name in exclude_set:
        continue
    for lambda_value in np.linspace(0,1, 21):
        protocol.append((name, np.round(lambda_value, 2)))

name, lambda_value = protocol[idx-1]
print(name)
print(lambda_value)

t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']

# generate both rdkit mol
if mode == 'forward':
    tautomer = neutromeratio.Tautomer(name=name, initial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles), nr_of_conformations=1)
    tautomer.perform_tautomer_transformation_forward()
    tautomer_atoms = tautomer.hybrid_atoms
    x0 = tautomer.hybrid_coords
elif mode == 'reverse':
    tautomer = neutromeratio.Tautomer(name=name, initial_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), nr_of_conformations=1)
    tautomer.perform_tautomer_transformation_reverse()
    tautomer_atoms = tautomer.hybrid_atoms
    x0 = tautomer.hybrid_coords
else:
    raise RuntimeError('No tautomer reaction direction was specified.')

os.makedirs(f"{base_path}/{name}", exist_ok=True)
# define the alchemical atoms
alchemical_atoms=[tautomer.hybrid_hydrogen_idx_at_lambda_1, tautomer.hybrid_hydrogen_idx_at_lambda_0]

print('Nr of atoms: {}'.format(len(tautomer.ligand_in_water_atoms)))

# extract hydrogen donor idx and hydrogen idx for from_mol
model = neutromeratio.ani.LinearAlchemicalSingleTopologyANI(alchemical_atoms=alchemical_atoms)
model = model.to(device)
torch.set_num_threads(2)

# perform initial sampling
energy_function = neutromeratio.ANI1_force_and_energy(
                                        model = model,
                                        atoms = tautomer_atoms,
                                        mol = None,
                                        )

for r in tautomer.ligand_restraints:
    energy_function.add_restraint(r)

for r in tautomer.hybrid_ligand_restraints:
    energy_function.add_restraint(r)

print(lambda_value)
energy_and_force = lambda x : energy_function.calculate_force(x, lambda_value)
langevin = neutromeratio.LangevinDynamics(atoms = tautomer_atoms,                            
                            energy_and_force = energy_and_force)

x0, e_history = energy_function.minimize(x0, maxiter=5000, lambda_value=lambda_value) 

equilibrium_samples, energies, ensemble_bias = langevin.run_dynamics(x0, n_steps=n_steps, 
stepsize=0.5 * unit.femtosecond, progress_bar=False)
   

# save equilibrium energy values 
f = open(f"{base_path}/{name}/{name}_lambda_{lambda_value:0.4f}_energy_in_vacuum_{mode}.csv", 'w+')
for e in energies[::20]:
    e_unitless = e / kT
    f.write('{}\n'.format(e_unitless))
f.close()

f = open(f"{base_path}/{name}/{name}_lambda_{lambda_value:0.4f}_restraint_bias_in_vacuum_{mode}.csv", 'w+')
for e in ensemble_bias[::20]:
    e_unitless = e / kT
    f.write('{}\n'.format(e_unitless))
f.close()


equilibrium_samples = [x.value_in_unit(unit.nanometer) for x in equilibrium_samples]
ani_traj = md.Trajectory(equilibrium_samples[::20], tautomer.hybrid_topology)
ani_traj.save(f"{base_path}/{name}/{name}_lambda_{lambda_value:0.4f}_in_vacuum_{mode}.dcd", force_overwrite=True)
