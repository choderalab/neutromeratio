from pymbar import MBAR
from openmmtools.constants import kB
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

temperature = 300 * unit.kelvin
kT = kB * temperature

# read in exp results, smiles and names
exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))

# name of the system
name = str(sys.argv[1])

# don't change - direction is fixed for all runs
#################
from_mol_tautomer_idx = 1
to_mol_tautomer_idx = 2

t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']

# generate both rdkit mol
mols = { 't1' : neutromeratio.generate_rdkit_mol(t1_smiles), 't2' : neutromeratio.generate_rdkit_mol(t2_smiles) }
from_mol = mols[f"t{from_mol_tautomer_idx}"]
to_mol = mols[f"t{to_mol_tautomer_idx}"]
ani_input = neutromeratio.from_mol_to_ani_input(from_mol)

# get tautomer transformation
tautomer_transformation = neutromeratio.get_tautomer_transformation(from_mol, to_mol)
neutromeratio.generate_hybrid_structure(ani_input, tautomer_transformation, neutromeratio.ani.ANI1_force_and_energy)
# define the alchemical atoms
alchemical_atoms=[tautomer_transformation['acceptor_hydrogen_idx'], tautomer_transformation['donor_hydrogen_idx']]

# 21 steps inclusive endpoints
# getting all the energies, snapshots and lambda values in lists
lambda_value = 0.0
energies = []
ani_trajs = []
lambdas = []
for _ in range(21):
    f_traj = f"/home/mwieder/Work/Projects/neutromeratio/data/equilibrium_sampling/{name}/{name}_lambda_{lambda_value:0.4f}.dcd"
    traj = md.load_dcd(f_traj, top=ani_input['hybrid_topology'])
    ani_trajs.append(traj)
    
    energy_file = open(f"/home/mwieder/Work/Projects/neutromeratio/data/equilibrium_sampling/{name}/{name}_lambda_{lambda_value:0.4f}_energy.csv", 'r')  
    tmp_e = []
    for e in energy_file:
        tmp_e.append(float(e))
    energy_file.close()
    
    energies.append(tmp_e)
    lambdas.append(lambda_value)
    lambda_value +=0.05

# plotting the energies of all equilibrium runs
for e in energies: 
    plt.plot(e, alpha=0.5)
plt.show()
plt.savefig(f"/home/mwieder/Work/Projects/neutromeratio/data/equilibrium_sampling/{name}/{name}_energy.png")

# generating energies for all lambda and all snapshots 
platform = 'cpu'
device = torch.device(platform)
model = neutromeratio.ani.LinearAlchemicalSingleTopologyANI(device=device, alchemical_atoms=alchemical_atoms, ani_input=ani_input)
model = model.to(device)
torch.set_num_threads(2)


energy_function = neutromeratio.ANI1_force_and_energy(device = device,
                                          model = model,
                                          atom_list = ani_input['hybrid_atoms'],
                                          platform = platform,
                                          tautomer_transformation = tautomer_transformation)
energy_function.restrain_acceptor = True
energy_function.restrain_donor = True

# TODO: equil time, thinning should be selected automatically, but here just hard-coded
equil = 500
thinning = 50
N_k = []

# form list of decorrelated snapshots from all lambda windows
snapshots = []
for traj in ani_trajs:
    new_snapshots = list(traj[equil::thinning].xyz * unit.nanometer)
    N_k.append(len(new_snapshots))
    snapshots.extend(new_snapshots)

# form u_kn matrix
K = len(lambdas)
N = len(snapshots)

u_kn = np.zeros((K, N))
for k in range(K):
    lamb = lambdas[k]
    energy_function.lambda_value = lamb
    for n in tqdm(range(N)):
        u_kn[k, n] = energy_function.calculate_energy(snapshots[n]) / kT

# pass u_kn matrix into pymbar
mbar = MBAR(u_kn, N_k)

# report
f_k = (mbar.f_k * kT).value_in_unit(unit.kilocalorie_per_mole)
deltaF_kcalmol = f_k[-1] - f_k[0]

f = open('/home/mwieder/Work/Projects/neutromeratio/data/equilibrium_sampling/energies.csv', 'a+')
f.write(f"{name}, {deltaF_kcalmol}\n")
f.close()
