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
from neutromeratio.constants import device, platform
import sys, os

# name of the system
idx = int(sys.argv[1])
# number of steps
n_steps = int(sys.argv[2])

mode = 'forward'

protocol = []
exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))
for name in exp_results:
    for lambda_value in np.linspace(0,1,11):
        protocol.append((name, lambda_value))

name, lambda_value = protocol[idx]
print(name)
print(lambda_value)

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
diameter_in_angstrom = 16
m = tautomer.add_droplet(tautomer.hybrid_topology, tautomer.hybrid_coords, diameter=diameter_in_angstrom * unit.angstrom)

# define the alchemical atoms
alchemical_atoms=[tautomer.hybrid_dummy_hydrogen, tautomer.hydrogen_idx]

# extract hydrogen donor idx and hydrogen idx for from_mol
model = neutromeratio.ani.LinearAlchemicalDualTopologyANI(alchemical_atoms=alchemical_atoms)
model = model.to(device)
torch.set_num_threads(2)

# perform initial sampling
energy_function = neutromeratio.ANI1_force_and_energy(
                                        model = model,
                                        atoms = tautomer.ligand_in_water_atoms,
                                        mol = tautomer.ligand_in_water_ase_mol,
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

print(lambda_value)
energy_and_force = lambda x : energy_function.calculate_force(x, lambda_value)
langevin = neutromeratio.LangevinDynamics(atoms = tautomer.ligand_in_water_atoms,                            
                            energy_and_force = energy_and_force)
x0 = np.array(tautomer.ligand_in_water_coordinates) * unit.angstrom
#x0 = energy_function.minimize(x0) # NOTE: No minimizing!

if not os.path.exists(f"/data/chodera/wiederm/equilibrium_sampling/{name}/"):
    os.makedirs(f"/data/chodera/wiederm/equilibrium_sampling/{name}/")

equilibrium_samples, energies, bias = langevin.run_dynamics(x0, n_steps=n_steps, stepsize=0.5 * unit.femtosecond, progress_bar=False)
    

# save equilibrium energy values 
f = open(f"/data/chodera/wiederm/equilibrium_sampling/{name}/{name}_lambda_{lambda_value:0.4f}_energy_in_droplet_{mode}.csv", 'w+')
for e in energies[::25]:
    f.write('{}\n'.format(e))
f.close()

f = open(f"/data/chodera/wiederm/equilibrium_sampling/{name}/{name}_lambda_{lambda_value:0.4f}_bias_in_droplet_{mode}.csv", 'w+')
for e in bias[::25]:
    f.write('{}\n'.format(e))
f.close()


equilibrium_samples = [x.value_in_unit(unit.nanometer) for x in equilibrium_samples]
ani_traj = md.Trajectory(equilibrium_samples[::25], tautomer.ligand_in_water_topology)
ani_traj.save(f"/data/chodera/wiederm/equilibrium_sampling/{name}/{name}_lambda_{lambda_value:0.4f}_in_droplet_{mode}.dcd", force_overwrite=True)