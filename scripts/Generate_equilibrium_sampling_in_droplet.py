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
from neutromeratio.constants import device, platform, kT
import sys, os
from tqdm import tqdm


exclude = ['molDWRow_1004', 'molDWRow_1110', 'molDWRow_1184', 'molDWRow_1185', 'molDWRow_1189', 'molDWRow_1262', 'molDWRow_1263',
'molDWRow_1267', 'molDWRow_1275', 'molDWRow_1279', 'molDWRow_1280', 'molDWRow_1282', 'molDWRow_1283', 'molDWRow_553',
'molDWRow_557', 'molDWRow_580', 'molDWRow_581', 'molDWRow_582', 'molDWRow_615', 'molDWRow_616', 'molDWRow_617',
'molDWRow_618', 'molDWRow_643', 'molDWRow_758', 'molDWRow_82', 'molDWRow_83', 'molDWRow_952', 'molDWRow_953',
'molDWRow_955', 'molDWRow_988', 'molDWRow_989', 'molDWRow_990', 'molDWRow_991', 'molDWRow_992']

# name of the system
idx = int(sys.argv[1])
# number of steps
n_steps = int(sys.argv[2])
# diameter
diameter_in_angstrom = int(sys.argv[3])
# where to write the results
base_path = str(sys.argv[4])

mode = 'forward'

protocol = []
exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))
for name in sorted(exp_results):
    if name in exclude:
        continue
    for lambda_value in np.linspace(0,1, 21):
        protocol.append((name, np.round(lambda_value, 2)))

name, lambda_value = protocol[idx-1]
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

os.makedirs(f"{base_path}/{name}", exist_ok=True)
m = tautomer.add_droplet(tautomer.hybrid_topology, 
                            tautomer.hybrid_coords, 
                            diameter=diameter_in_angstrom * unit.angstrom,
                            restrain_hydrogens=True,
                            file=f"{base_path}/{name}/{name}_in_droplet_{mode}.pdb")

# define the alchemical atoms
alchemical_atoms=[tautomer.hybrid_hydrogen_idx_at_lambda_1, tautomer.hybrid_hydrogen_idx_at_lambda_0]

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

print(lambda_value)
energy_and_force = lambda x : energy_function.calculate_force(x, lambda_value)
langevin = neutromeratio.LangevinDynamics(atoms = tautomer.ligand_in_water_atoms,                            
                            energy_and_force = energy_and_force)

x0 = tautomer.ligand_in_water_coordinates
x0, e_history = energy_function.minimize(x0, maxiter=1000, lambda_value=lambda_value) 

n_steps_junk = n_steps/10

equilibrium_samples_global = []
energies_global = []
bias_global = []

for n_steps in [n_steps_junk] *10:
    equilibrium_samples, energies, bias = langevin.run_dynamics(x0, n_steps=round(n_steps), stepsize=0.5 * unit.femtosecond, progress_bar=False)
    
    # set new x0
    x0 = equilibrium_samples[-1]

    # add to global list
    equilibrium_samples_global += equilibrium_samples
    energies_global += energies
    bias_global += bias

    # save equilibrium energy values 
    f = open(f"{base_path}/{name}/{name}_lambda_{lambda_value:0.4f}_energy_in_droplet_{mode}.csv", 'w+')
    for e in energies_global[::20]:
        e_unitless = e / kT
        f.write('{}\n'.format(e_unitless))
    f.close()

    f = open(f"{base_path}/{name}/{name}_lambda_{lambda_value:0.4f}_bias_in_droplet_{mode}.csv", 'w+')
    for e in bias_global[::20]:
        e_unitless = e / kT
        f.write('{}\n'.format(e_unitless))
    f.close()


    equilibrium_samples_in_nm = [x.value_in_unit(unit.nanometer) for x in equilibrium_samples_global]
    ani_traj = md.Trajectory(equilibrium_samples_in_nm[::20], tautomer.ligand_in_water_topology)
    ani_traj.save(f"{base_path}/{name}/{name}_lambda_{lambda_value:0.4f}_in_droplet_{mode}.dcd", force_overwrite=True)

