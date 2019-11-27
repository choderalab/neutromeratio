#%%
import neutromeratio
from openmmtools.constants import kB
from simtk import unit
import numpy as np
import pickle
import mdtraj as md
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import torch
from neutromeratio.parameter_gradients import FreeEnergyCalculator
from neutromeratio.constants import kT, device, exclude_set
from glob import glob

def parse_lambda_from_dcd_filename(dcd_filename, env):
    return float(dcd_filename[:dcd_filename.find(f"_in_{env}")].split('_')[-1])


# job idx
idx = int(1)
# where to write the results
base_path = str('/home/mwieder/')
env = str('droplet')
diameter_in_angstrom = int(18)
#######################

mode = 'forward'

# read in exp results, smiles and names
exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))

# name of the system
protocoll = []
for name in sorted(exp_results):
    if name in exclude_set:
        continue
    protocoll.append(name)

name = protocoll[idx-1]
print(name)

# don't change - direction is fixed for all runs
#################
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

if env == 'droplet':
    tautomer.add_droplet(tautomer.hybrid_topology, 
                                tautomer.hybrid_coords, 
                                diameter=diameter_in_angstrom * unit.angstrom,
                                restrain_hydrogens=True,
                                file=f"{base_path}/{name}/{name}_in_droplet_{mode}.pdb")
    print('Nr of atoms: {}'.format(len(tautomer.ligand_in_water_atoms)))
    atoms = tautomer.ligand_in_water_atoms
    top = tautomer.ligand_in_water_topology
else:
    atoms = tautomer.hybrid_atoms
    top = tautomer.hybrid_topology

# define the alchemical atoms
alchemical_atoms=[tautomer.hybrid_hydrogen_idx_at_lambda_1, tautomer.hybrid_hydrogen_idx_at_lambda_0]


# extract hydrogen donor idx and hydrogen idx for from_mol
model = neutromeratio.ani.LinearAlchemicalDualTopologyANI(alchemical_atoms=alchemical_atoms, adventure_mode=True)
model = model.to(device)
torch.set_num_threads(2)

# perform initial sampling
energy_function = neutromeratio.ANI1_force_and_energy(
                                        model = model,
                                        atoms = atoms,
                                        mol = None,
                                        )


for r in tautomer.ligand_restraints:
    energy_function.add_restraint(r)

for r in tautomer.hybrid_ligand_restraints:
    energy_function.add_restraint(r)

if env == 'droplet':

    tautomer.add_COM_for_hybrid_ligand(np.array([diameter_in_angstrom/2, diameter_in_angstrom/2, diameter_in_angstrom/2]) * unit.angstrom)

    for r in tautomer.solvent_restraints:
        energy_function.add_restraint(r)

    for r in tautomer.com_restraints:
        energy_function.add_restraint(r)


# get steps inclusive endpoints
# and lambda values in list
dcds = glob(f"{base_path}/{name}/*.dcd")

lambdas = []
ani_trajs = {}
energies = []

for dcd_filename in dcds:
    lam = parse_lambda_from_dcd_filename(dcd_filename, env)
    lambdas.append(lam)
    traj = md.load_dcd(dcd_filename, top=top)
    ani_trajs[lam] = list(traj[0:200:10].xyz * unit.nanometer)
    f = open(f"{base_path}/{name}/{name}_lambda_{lam:0.4f}_energy_in_{env}_{mode}.csv", 'r')  
    tmp_e = []
    for e in f:
        tmp_e.append(float(e))
    f.close()
    energies.append(np.array(tmp_e))

# %%
def calculate_stddev(snapshots):
    lambda0_e_b_stddev = [energy_function.calculate_energy(x, lambda_value=0.0) for x in tqdm(snapshots)]
    lambda1_e_b_stddev = [energy_function.calculate_energy(x, lambda_value=1.0) for x in tqdm(snapshots)]

    # extract endpoint stddev
    lambda0_stddev = [stddev/kT for stddev in [e_b_stddev[2] for e_b_stddev in lambda0_e_b_stddev]]
    lambda1_stddev = [stddev/kT for stddev in [e_b_stddev[2] for e_b_stddev in lambda1_e_b_stddev]]
    return np.array(lambda0_stddev), np.array(lambda1_stddev)

def compute_linear_penalty(current_stddev, n_atoms):
    per_atom_thresh = 0.5 * unit.kilojoule_per_mole
    total_thresh = per_atom_thresh * n_atoms
    linear_penalty = np.maximum(0, current_stddev - (total_thresh/kT))
    return linear_penalty

def compute_last_valid_ind(linear_penalty):
    if linear_penalty.sum() > 0:
        last_valid_ind = np.argmax(np.cumsum(linear_penalty) > 0) -1
    else:
        last_valid_ind = len(linear_penalty) -1
    return last_valid_ind

# %%
n_atoms = len(atoms)
last_valid_inds = {}
for lam in ani_trajs:
    lambda0_stddev, lambda1_stddev = calculate_stddev(ani_trajs[lam])
    current_stddev = (1 - lam) * lambda0_stddev + lam * lambda1_stddev
    print(current_stddev)
    linear_penalty = compute_linear_penalty(current_stddev, n_atoms)
    last_valid_ind = compute_last_valid_ind(linear_penalty)
    print(last_valid_ind)
    last_valid_inds[lam] = last_valid_ind


# %%
print(last_valid_inds)

lambdas_with_usable_samples = []
for lam in sorted(list(last_valid_inds.keys())):
    if last_valid_inds[lam] > 5:
        lambdas_with_usable_samples.append(lam)
lambdas_with_usable_samples

# %%
snapshots = []
N_k = []

max_n_snapshots_per_state = 10

for lam in lambdas_with_usable_samples:
    traj = ani_trajs[lam][0:last_valid_inds[lam]]
    further_thinning = 1
    if len(traj) > max_n_snapshots_per_state:
        further_thinning = int(len(traj) / max_n_snapshots_per_state)
    new_snapshots = traj[::further_thinning]
    snapshots.extend(new_snapshots)
    N_k.append(len(new_snapshots))

N = len(snapshots)
N_k, N

# %%
