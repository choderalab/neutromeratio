import operator
import neutromeratio
from openmmtools.constants import kB
from simtk import unit
import numpy as np
import pickle
import mdtraj as md
import matplotlib.pyplot as plt
import sys
import torch
from tqdm import tqdm
from neutromeratio.parameter_gradients import FreeEnergyCalculator
from neutromeratio.constants import kT, device, exclude_set_ANI, mols_with_charge
from glob import glob
import seaborn as sns
import os
import torchani

latest_checkpoint = 'latest.pt'
# read in exp results, smiles and names
exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
def parse_lambda_from_dcd_filename(dcd_filename, env):
    l = dcd_filename[:dcd_filename.find(f"_energy_in_{env}")].split('_')
    lam = l[-3]
    return float(lam)

#######################
thinning = 50
per_atom_stddev_threshold = 10.0 * unit.kilojoule_per_mole 
#######################

sns.set_context('paper')
sns.set(color_codes=True)

env = 'vacuum'
name = 'SAMPLmol2'
base_path = f"./data/"

if name in exclude_set_ANI + mols_with_charge:
    raise RuntimeError(f"{name} is part of the list of excluded molecules. Aborting")

t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']
print(f"Experimental free energy difference: {exp_results[name]['energy']} kcal/mol")

#######################
#######################

t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles)
tautomer = tautomers[0]
tautomer.perform_tautomer_transformation()

atoms = tautomer.hybrid_atoms
top = tautomer.hybrid_topology

# define the alchemical atoms
alchemical_atoms = [tautomer.hybrid_hydrogen_idx_at_lambda_1, tautomer.hybrid_hydrogen_idx_at_lambda_0]

# extract hydrogen donor idx and hydrogen idx for from_mol
model = neutromeratio.ani.LinearAlchemicalSingleTopologyANI(alchemical_atoms=alchemical_atoms)

model = model.to(device)
torch.set_num_threads(1)

# perform initial sampling
energy_function = neutromeratio.ANI1_force_and_energy(
    model=model,
    atoms=atoms,
    mol=None,
    per_atom_thresh=10.4 * unit.kilojoule_per_mole,
    adventure_mode=True
)

# add restraints
for r in tautomer.ligand_restraints:
    energy_function.add_restraint_to_lambda_protocol(r)
for r in tautomer.hybrid_ligand_restraints:
    energy_function.add_restraint_to_lambda_protocol(r)

# get steps inclusive endpoints
# and lambda values in list
dcds = glob(f"{base_path}/{name}/*.dcd")

lambdas = []
ani_trajs = []
energies = []

# read in all the frames from the trajectories
for dcd_filename in dcds:
    lam = parse_lambda_from_dcd_filename(dcd_filename, env)
    lambdas.append(lam)
    traj = md.load_dcd(dcd_filename, top=top)[::thinning]
    print(f"Nr of frames in trajectory: {len(traj)}")
    ani_trajs.append(traj)
    f = open(f"{base_path}/{name}/{name}_lambda_{lam:0.4f}_energy_in_{env}.csv", 'r')
    energies.append(np.array([float(e) for e in f][::thinning]))
    f.close()

# calculate free energy in kT
fec = FreeEnergyCalculator(ani_model=energy_function,
                            ani_trajs=ani_trajs,
                            potential_energy_trajs=energies,
                            lambdas=lambdas,
                            n_atoms=len(atoms),
                            max_snapshots_per_window=-1,
                            per_atom_thresh=per_atom_stddev_threshold)

# defining neural networks
nn = model.neural_networks
aev_dim = model.aev_computer.aev_length
# define which layer should be modified -- currently the last one
layer = 6
# take only a single network from the ensemble of 8
single_nn = model.neural_networks[0]

# set up minimizer for weights
AdamW = torchani.optim.AdamW([
    {'params' : [single_nn.C[layer].weight], 'weight_decay': 0.000001},
    {'params' : [single_nn.H[layer].weight], 'weight_decay': 0.000001},
    {'params' : [single_nn.O[layer].weight], 'weight_decay': 0.000001},
    {'params' : [single_nn.N[layer].weight], 'weight_decay': 0.000001},
])

# set up minimizer for bias
SGD = torch.optim.SGD([
    {'params' : [single_nn.C[layer].bias]},
    {'params' : [single_nn.H[layer].bias]},
    {'params' : [single_nn.O[layer].bias]},
    {'params' : [single_nn.N[layer].bias]},
], lr=1e-3)


AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)
SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=100, threshold=0)

# save checkpoint
if os.path.isfile(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint)
    nn.load_state_dict(checkpoint['nn'])
    AdamW.load_state_dict(checkpoint['AdamW'])
    SGD.load_state_dict(checkpoint['SGD'])
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
    SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])


# calculate free energy
def validate():
    'return free energy in kT'
    
    #return torch.tensor([5.0], device=device)
    if flipped:
        deltaF = fec.compute_free_energy_difference() * -1.
    else:
        deltaF = fec.compute_free_energy_difference()
    return deltaF

# return the experimental value
def experimental_value():
    e_in_kT = (exp_results[name]['energy'] * unit.kilocalorie_per_mole)/kT
    return torch.tensor([e_in_kT], device=device)


print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
max_epochs = 100
early_stopping_learning_rate = 1.0E-5
best_model_checkpoint = 'best.pt'

for _ in tqdm(range(AdamW_scheduler.last_epoch + 1, max_epochs)):
    rmse = torch.sqrt((validate() - experimental_value())**2)
    print(rmse)
            
     # checkpoint
    if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
        torch.save(nn.state_dict(), best_model_checkpoint)

    # define the stepsize -- very conservative
    AdamW_scheduler.step(rmse/10)
    SGD_scheduler.step(rmse/10)
    loss = rmse

    AdamW.zero_grad()
    SGD.zero_grad()
    loss.backward()
    AdamW.step()
    SGD.step()

torch.save({
    'nn': nn.state_dict(),
    'AdamW': AdamW.state_dict(),
    'SGD': SGD.state_dict(),
    'AdamW_scheduler': AdamW_scheduler.state_dict(),
    'SGD_scheduler': SGD_scheduler.state_dict(),
}, latest_checkpoint)