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
exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']

# generate both rdkit mol
mols = { 't1' : neutromeratio.generate_rdkit_mol(t1_smiles), 't2' : neutromeratio.generate_rdkit_mol(t2_smiles) }
from_mol = mols['t1']
to_mol = mols['t2']

# set model
model = torchani.models.ANI1ccx()
model = model.to(device)
torch.set_num_threads(2)
e = []

# calculate energy using both structures and pure ANI1ccx
for tautomer in [from_mol, to_mol]: 
    ani_input = neutromeratio.from_mol_to_ani_input(tautomer, nr_of_conf=1)
    energy_function = neutromeratio.ANI1_force_and_energy(
                                            model = model,
                                            atoms = ani_input['ligand_atoms'],
                                            use_pure_ani1ccx = True
                                        )
    # minimize
    energy_function.minimize(ani_input, hybrid=False)

    x0 = np.array(ani_input['ligand_coords'][0]) * unit.angstrom
    e.append(energy_function.calculate_energy(x0))
e_diff = (e[1] - e[0])
f = open('/home/mwieder/Work/Projects/neutromeratio/data/equilibrium_sampling/energies.csv', 'a+')
f.write(f"{name}, {e_diff}\n")
f.close()
