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
from neutromeratio.constants import platform, device, exclude_set

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

# extract smiles
t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']

entropy_correction = pickle.load(open('../data/results/entropy_correction.pickle', 'rb'))
t1_entropy_correction = entropy_correction[name]['t1-entropy-correction [kT]']
t2_entropy_correction = entropy_correction[name]['t2-entropy-correction [kT]']

tautomer = neutromeratio.Tautomer(name=name, initial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), 
                                    final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles), 
                                    nr_of_conformations=100, 
                                    enforceChirality=False)
tautomer.perform_tautomer_transformation_forward()

# set model
model = torchani.models.ANI1ccx()
model = model.to(device)
torch.set_num_threads(2)


# t1
# calculate energy using both structures and pure ANI1ccx
energy_function = neutromeratio.ANI1_force_and_energy(
                                        model = model,
                                        mol = tautomer.initial_state_ase_mol,
                                        atoms = tautomer.initial_state_ligand_atoms,
                                        use_pure_ani1ccx = True)

t1_e = []
t1_g = []
for coords in tautomer.initial_state_ligand_coords:
    # minimize
    minimized_coords, _ = energy_function.minimize(coords)
    # calculate electronic single point energy
    e = energy_function.calculate_energy(minimized_coords)
    # calculate Gibb's free energy
    try:
        thermochemistry_correction = energy_function.get_thermo_correction(minimized_coords)  
    except ValueError:
        print('Imaginary frequencies present - found transition state.')
        continue

    g = neutromeratio.reduced_pot(e) + neutromeratio.reduced_pot(thermochemistry_correction) + t1_entropy_correction
    
    t1_e.append(e)
    t1_g.append(g)

# t2
# calculate energy using both structures and pure ANI1ccx
energy_function = neutromeratio.ANI1_force_and_energy(
                                        model = model,
                                        mol = tautomer.final_state_ase_mol,
                                        atoms = tautomer.final_state_ligand_atoms,
                                        use_pure_ani1ccx = True)

t2_e = []
t2_g = []
for coords in tautomer.final_state_ligand_coords:
    # minimize
    minimized_coords, _ = energy_function.minimize(coords)
    # calculate electronic single point energy
    e = energy_function.calculate_energy(minimized_coords)
    # calculate Gibb's free energy
    try:
        thermochemistry_correction = energy_function.get_thermo_correction(minimized_coords)  
    except ValueError:
        print('Imaginary frequencies present - found transition state.')
        continue

    g = neutromeratio.reduced_pot(e) + neutromeratio.reduced_pot(thermochemistry_correction) + t2_entropy_correction
    
    t2_e.append(e)
    t2_g.append(g)
       
# random aka the first dE and dG energy difference
e_diff = t2_e[0] - t1_e[0]
g_diff = t2_g[0] - t1_g[0]
# write dE
f = open('/home/mwieder/Work/Projects/neutromeratio/data/results/ANI1ccx_vacuum_random_minimum_dE.csv', 'a+')
f.write(f"{name}, {e_diff}\n")
f.close()
# write dG
f = open('/home/mwieder/Work/Projects/neutromeratio/data/results/ANI1ccx_vacuum_random_minimum_dG.csv', 'a+')
f.write(f"{name}, {g_diff}\n")
f.close()



# 'global' minimum dE and dG energy difference
e_diff = min(t2_e) - min(t1_e)
g_diff = min(t2_g) - min(t1_g)
# write dE
f = open('/home/mwieder/Work/Projects/neutromeratio/data/results/ANI1ccx_vacuum_global_minimum_dE.csv', 'a+')
f.write(f"{name}, {e_diff}\n")
f.close()
# write dG
f = open('/home/mwieder/Work/Projects/neutromeratio/data/results/ANI1ccx_vacuum_global_minimum_dG.csv', 'a+')
f.write(f"{name}, {g_diff}\n")
f.close()
