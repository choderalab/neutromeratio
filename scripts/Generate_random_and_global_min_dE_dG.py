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
from neutromeratio.constants import platform, device, exclude_set_ANI, mols_with_charge


entropy_correction = True
if entropy_correction:
    ending = '_with_entropy_correction'
else:
    ending = ''
# job idx
idx = int(sys.argv[1])

# read in exp results, smiles and names
exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))

# name of the system
protocoll = []
for name in sorted(exp_results):
    if name in exclude_set_ANI + mols_with_charge:
        continue
    protocoll.append(name)

name = protocoll[idx-1]
print(name)

# extract smiles
t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']

t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name,
                                                                t1_smiles,
                                                                t2_smiles,
                                                                nr_of_conformations=10,
                                                                enforceChirality=True)
tautomer = tautomers[0]
tautomer.perform_tautomer_transformation()

# set model
model = neutromeratio.ani.PureANI1ccx()  
model = model.to(device)
torch.set_num_threads(2)


# t1
# calculate energy using both structures and pure ANI1ccx
energy_function = neutromeratio.ANI1_force_and_energy(
                                        model = model,
                                        mol = tautomer.initial_state_ase_mol,
                                        atoms = tautomer.initial_state_ligand_atoms,
                                        )

t1_e = []
t1_g = []
for coords in tautomer.initial_state_ligand_coords:
    # minimize
    minimized_coords, _ = energy_function.minimize(coords)
    # calculate electronic single point energy
    e, _, __, ___ = energy_function.calculate_energy(minimized_coords)
    # calculate Gibb's free energy
    try:
        thermochemistry_correction = energy_function.get_thermo_correction(minimized_coords)  
    except ValueError:
        print('Imaginary frequencies present - found transition state.')
        continue

    g = neutromeratio.reduced_pot(e) + neutromeratio.reduced_pot(thermochemistry_correction)
    e = neutromeratio.reduced_pot(e)

    if entropy_correction:
        g += neutromeratio.reduced_pot(tautomer.initial_state_entropy_correction)
        e += neutromeratio.reduced_pot(tautomer.final_state_entropy_correction)

    
  

    t1_e.append(e)
    t1_g.append(g)

# t2
# calculate energy using both structures and pure ANI1ccx
energy_function = neutromeratio.ANI1_force_and_energy(
                                        model = model,
                                        mol = tautomer.final_state_ase_mol,
                                        atoms = tautomer.final_state_ligand_atoms,
                                        )

t2_e = []
t2_g = []
for coords in tautomer.final_state_ligand_coords:
    # minimize
    minimized_coords, _ = energy_function.minimize(coords)
    # calculate electronic single point energy
    e, _, __, ___ = energy_function.calculate_energy(minimized_coords)
    # calculate Gibb's free energy
    try:
        thermochemistry_correction = energy_function.get_thermo_correction(minimized_coords)  
    except ValueError:
        print('Imaginary frequencies present - found transition state.')
        continue

    e = neutromeratio.reduced_pot(e)
    g = neutromeratio.reduced_pot(e) + neutromeratio.reduced_pot(thermochemistry_correction)
    if entropy_correction:
        g += neutromeratio.reduced_pot(tautomer.final_state_entropy_correction)
        e += neutromeratio.reduced_pot(tautomer.final_state_entropy_correction)
   
    t2_e.append(e)
    t2_g.append(g)
       
# random aka the first dE and dG energy difference
e_diff = t2_e[0] - t1_e[0]
g_diff = t2_g[0] - t1_g[0]
# write dE
f = open(f"/home/mwieder/Work/Projects/neutromeratio/data/results/ANI1ccx_vacuum_random_minimum_dE_only_one_stereo{ending}.csv", 'a+')
f.write(f"{name}, {e_diff}\n")
f.close()
# write dG
f = open(f"/home/mwieder/Work/Projects/neutromeratio/data/results/ANI1ccx_vacuum_random_minimum_dG_only_one_stereo{ending}.csv", 'a+')
f.write(f"{name}, {g_diff}\n")
f.close()



# 'global' minimum dE and dG energy difference
e_diff = min(t2_e) - min(t1_e)
g_diff = min(t2_g) - min(t1_g)
# write dE
f = open(f"/home/mwieder/Work/Projects/neutromeratio/data/results/ANI1ccx_vacuum_global_minimum_dE_only_one_stereo{ending}.csv", 'a+')
f.write(f"{name}, {e_diff}\n")
f.close()
# write dG
f = open(f"/home/mwieder/Work/Projects/neutromeratio/data/results/ANI1ccx_vacuum_global_minimum_dG_only_one_stereo{ending}.csv", 'a+')
f.write(f"{name}, {g_diff}\n")
f.close()
