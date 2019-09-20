import neutromeratio
from openmmtools.constants import kB
from simtk import unit
import numpy as np
import pickle
import mdtraj as md
import matplotlib.pyplot as plt
import sys
import torch
from neutromeratio.parameter_gradients import FreeEnergyCalculator

#######################
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
ani_input = neutromeratio.from_mol_to_ani_input(from_mol, nr_of_conf=1)

# get tautomer transformation
tautomer_transformation = neutromeratio.get_tautomer_transformation(from_mol, to_mol)
neutromeratio.generate_hybrid_structure(ani_input, tautomer_transformation)

# 20 steps inclusive endpoints
# getting all the energies, snapshots and lambda values in lists
# NOTE: This will be changed in the future
lambda_value = 0.0
for _ in range(20):
    f_traj = f"/home/mwieder/Work/Projects/neutromeratio/data/equilibrium_sampling/{name}/{name}_lambda_{lambda_value:0.4f}.dcd"
    traj = md.load_dcd(f_traj[::10], top=ani_input['hybrid_topology'])
    traj.save(f"../data/equilibrium_sampling/{name}/{name}_lambda_{lambda_value:0.4f}_thinned.dcd", force_overwrite=True)
    lambda_value +=0.05

