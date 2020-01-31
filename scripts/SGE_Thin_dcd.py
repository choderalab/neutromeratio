import neutromeratio
from openmmtools.constants import kB
from simtk import unit
import numpy as np
import pickle
import mdtraj as md
import matplotlib.pyplot as plt
import sys
import torch
from neutromeratio.constants import exclude_set

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

base = '/data/shared/projects/neutromeratio/data/equilibrium_sampling/waterbox-18A/'
pdb_file = f"{base}/{name}/{name}_in_droplet_forward.pdb"
lambda_value = 0.0
for _ in range(21):
    f_traj = f"{name}/{name}_lambda_{lambda_value:0.4f}.dcd"
    traj = md.load_dcd(f_traj, top=pdb_file)
    traj[::10].save(f"{base}/{name}/{name}_lambda_{lambda_value:0.4f}_thinned.dcd", force_overwrite=True)
    lambda_value +=0.05


