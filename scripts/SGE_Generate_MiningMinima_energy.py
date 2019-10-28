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
import random, sys, os

exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))

# name of the system
name = str(sys.argv[1])
torch.set_num_threads(2)

t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']

tautomer = neutromeratio.Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles), nr_of_conformations=100)
tautomer.perform_tautomer_transformation_forward()

confs_traj, mining_min_e, minimum_energies = tautomer.generate_mining_minima_structures()

#mkdir, write confs and structure
os.makedirs(f"/home/mwieder/Work/Projects/neutromeratio/data/mining_minima/{name}", exist_ok=True)
confs_traj[0].save_dcd(f"/home/mwieder/Work/Projects/neutromeratio/data/mining_minima/{name}/mm_confs_t1.dcd", force_overwrite=True)
confs_traj[0].save_pdb(f"/home/mwieder/Work/Projects/neutromeratio/data/mining_minima/{name}/mm_confs_t1.pdb", force_overwrite=True)
confs_traj[1].save_dcd(f"/home/mwieder/Work/Projects/neutromeratio/data/mining_minima/{name}/mm_confs_t2.dcd", force_overwrite=True)
confs_traj[1].save_pdb(f"/home/mwieder/Work/Projects/neutromeratio/data/mining_minima/{name}/mm_confs_t2.pdb", force_overwrite=True)

# write minimum energies
f = open(f"/home/mwieder/Work/Projects/neutromeratio/data/mining_minima/{name}/mm_confs_e_t1.csv", "w+")
for e in minimum_energies[0]:
    f.write(f"{neutromeratio.reduced_pot(e)}\n")
f.close()
# write minimum energies
f = open(f"/home/mwieder/Work/Projects/neutromeratio/data/mining_minima/{name}/mm_confs_e_t2.csv", "w+")
for e in minimum_energies[1]:
    f.write(f"{neutromeratio.reduced_pot(e)}\n")
f.close()

# write results
f = open('/home/mwieder/Work/Projects/neutromeratio/data/mining_minima/MM_ANIccx_vacuum.csv', 'a+')
f.write(f"{name}, {neutromeratio.reduced_pot(mining_min_e)}, {str(confs_traj[0].n_frames)}, {str(confs_traj[1].n_frames)}\n")
f.close()
