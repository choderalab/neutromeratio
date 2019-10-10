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

exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))

# name of the system
name = str(sys.argv[1])

from_mol_tautomer_idx = 1
to_mol_tautomer_idx = 2

t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']

tautomer = neutromeratio.Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles), nr_of_conformations=20)
tautomer.perform_tautomer_transformation_forward()

confs_traj, e = tautomer.generate_mining_minima_structures()

f = open('/home/mwieder/Work/Projects/neutromeratio/data/results/ANI1ccx_vacuum_MM_kcal.csv', 'a+')
f.write(f"{name}, {e}\n")
f.close()
