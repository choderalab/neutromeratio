import neutromeratio
from openmmtools.constants import kB
from neutromeratio.constants import exclude_set
from simtk import unit
import numpy as np
import pickle
import mdtraj as md
import torchani
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import random, sys, os

rmsd_threshold = 0.1 # Angstrom
# job idx
idx = int(sys.argv[1])

# read in exp results, smiles and names
exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))

# name of the system
protocoll = []
for name in sorted(exp_results):
    if name in exclude_set:
        continue
    protocoll.append(name)

name = protocoll[idx-1]
print(name)

torch.set_num_threads(2)

t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']

tautomer = neutromeratio.Tautomer(name=name, initial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles), nr_of_conformations=100)
tautomer.perform_tautomer_transformation_forward()

print('Treshold used for RMSD filtering: {}'.format(rmsd_threshold))
confs_traj, mining_min_e, minimum_energies, all_energies, all_conformations = tautomer.generate_mining_minima_structures(rmsd_threshold=rmsd_threshold, 
                                                                                        include_entropy_correction=True)
d = {'t1-energies' : all_energies[0], 't2-energies' : all_energies[1], 't1-confs' : all_conformations[0], , 't2-confs' : all_conformations[1]}

#mkdir, write confs and structure
base = "/home/mwieder/Work/Projects/neutromeratio/data/mining_minima"
os.makedirs(f"{base}/{name}", exist_ok=True)
# all confs and energies
with open(f"{base}/{name}/all_confs_energies.pickle", 'wb') as f:
    pickle.dump(d, f)
    
confs_traj[0].save_dcd(f"{base}/{name}/mm_confs_t1.dcd", force_overwrite=True)
confs_traj[0].save_pdb(f"{base}/{name}/mm_confs_t1.pdb", force_overwrite=True)
confs_traj[1].save_dcd(f"{base}/{name}/mm_confs_t2.dcd", force_overwrite=True)
confs_traj[1].save_pdb(f"{base}/{name}/mm_confs_t2.pdb", force_overwrite=True)

# write minimum energies
with open(f"{base}/{name}/mm_confs_e_t1.csv", "w+") as f:
    for e in minimum_energies[0]:
        f.write(f"{neutromeratio.reduced_pot(e)}\n")
# write minimum energies
with open(f"{base}/{name}/mm_confs_e_t2.csv", "w+") as f:
    for e in minimum_energies[1]:
        f.write(f"{neutromeratio.reduced_pot(e)}\n")

# write results
with open(f"{base}/MM_ANI1ccx_vacuum.csv", 'a+') as f:
    f.write(f"{name}, {neutromeratio.reduced_pot(mining_min_e)}, {str(confs_traj[0].n_frames)}, {str(confs_traj[1].n_frames)}\n")
