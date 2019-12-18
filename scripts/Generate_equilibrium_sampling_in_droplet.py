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
from tqdm import tqdm


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
    if name in exclude_set:
        continue
    for lambda_value in np.linspace(0,1,21):
        protocol.append((name, np.round(lambda_value, 2)))

name, lambda_value = protocol[idx-1]
print(name)
print(lambda_value)

t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']

# generate both rdkit mol
tautomer = neutromeratio.Tautomer(name=name, initial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles), nr_of_conformations=20)
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
                            restrain_hydrogen_bonds=True,
                            restrain_hydrogen_angles=False,
                            file=f"{base_path}/{name}/{name}_in_droplet_{mode}.pdb")

# define the alchemical atoms
alchemical_atoms=[tautomer.hybrid_hydrogen_idx_at_lambda_1, tautomer.hybrid_hydrogen_idx_at_lambda_0]

print('Nr of atoms: {}'.format(len(tautomer.ligand_in_water_atoms)))


# extract hydrogen donor idx and hydrogen idx for from_mol
model = neutromeratio.ani.LinearAlchemicalDualTopologyANI(alchemical_atoms=alchemical_atoms)
model = model.to(device)
torch.set_num_threads(1)

# perform initial sampling
energy_function = neutromeratio.ANI1_force_and_energy(
                                        model = model,
                                        atoms = tautomer.ligand_in_water_atoms,
                                        mol = None,
                                        per_atom_thresh=0.4 * unit.kilojoule_per_mole,
                                        adventure_mode=True
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
x0, e_history = energy_function.minimize(x0, maxiter=200, lambda_value=lambda_value) 
n_steps_junk = int(n_steps/10)

equilibrium_samples_global = []
energies_global = []
bias_global = []
stddev_global = []
penalty_global = []

for n_steps in [n_steps_junk] *10:
    equilibrium_samples, energies, bias, stddev, penalty = langevin.run_dynamics(x0, 
                                                                n_steps=round(n_steps), 
                                                                stepsize=0.5 * unit.femtosecond, 
                                                                progress_bar=False)
    
    # set new x0
    x0 = equilibrium_samples[-1]

    # add to global list
    equilibrium_samples_global += equilibrium_samples
    energies_global += energies
    bias_global += bias
    stddev_global += stddev
    penalty_global += penalty

    # save equilibrium energy values 
    for global_list, poperty_name in zip([energies_global, stddev_global, bias_global, penalty_global], ['energy', 'stddev', 'bias', 'penalty']):
        f = open(f"{base_path}/{name}/{name}_lambda_{lambda_value:0.4f}_{poperty_name}_in_droplet_{mode}.csv", 'w+')
        for e in global_list[::20]:
            e_unitless = e / kT
            f.write('{}\n'.format(e_unitless))
        f.close()   

    equilibrium_samples_in_nm = [x.value_in_unit(unit.nanometer) for x in equilibrium_samples_global]
    ani_traj = md.Trajectory(equilibrium_samples_in_nm[::20], tautomer.ligand_in_water_topology)
    ani_traj.save(f"{base_path}/{name}/{name}_lambda_{lambda_value:0.4f}_in_droplet_{mode}.dcd", force_overwrite=True)
