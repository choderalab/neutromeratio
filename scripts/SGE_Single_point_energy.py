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
from neutromeratio.constants import platform, device

# name of the system
name = str(sys.argv[1])

# extract smiles
exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))
t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']

tautomer = neutromeratio.Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles), nr_of_conformations=100)
tautomer.perform_tautomer_transformation_forward()

# set model
model = torchani.models.ANI1ccx()
model = model.to(device)
torch.set_num_threads(2)
e = []

# t1
t1_e = []
# calculate energy using both structures and pure ANI1ccx
energy_function = neutromeratio.ANI1_force_and_energy(
                                        model = model,
                                        mol = tautomer.intial_state_ase_mol,
                                        atoms = tautomer.intial_state_ligand_atoms,
                                        use_pure_ani1ccx = True)

for cor in tautomer.intial_state_ligand_coords:
    # minimize
    energy_function.minimize(cor)

    x0 = np.array(cor) * unit.angstrom
    t1_e.append(energy_function.calculate_energy(x0))
    
# t2
t2_e = []
# calculate energy using both structures and pure ANI1ccx
energy_function = neutromeratio.ANI1_force_and_energy(
                                        model = model,
                                        mol = tautomer.final_state_ase_mol,
                                        atoms = tautomer.final_state_ligand_atoms,
                                        use_pure_ani1ccx = True)

for cor in tautomer.final_state_ligand_coords:
    # minimize
    energy_function.minimize(cor)

    x0 = np.array(cor) * unit.angstrom
    t2_e.append(energy_function.calculate_energy(x0))
       
    
e_diff = neutromeratio.reduced_pot(min(t2_e) - min(t1_e))
print(e_diff)
#f = open('/home/mwieder/Work/Projects/neutromeratio/data/diff_single_point_minimized_energiesV2.csv', 'a+')
#f.write(f"{name}, {e_diff}\n")
#f.close()
