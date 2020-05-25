import operator
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
from neutromeratio.constants import kT, device, exclude_set_ANI, mols_with_charge
from glob import glob

#######################
#######################
# job idx
idx = int(sys.argv[1])
# where to write the results
base_path = str(sys.argv[2])
#######################
#######################
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
fec_list, model = neutromeratio.analysis.setup_mbar([name], base_path, 100)

fec = fec_list[0]

DeltaF_ji, dDeltaF_ji = fec.end_state_free_energy_difference
if fec.flipped:
    DeltaF_ji *= -1

print(fec.end_state_free_energy_difference)
f = open(f"{base_path}/results_in_kT.csv", 'a+')
f.write(f"{name}, {DeltaF_ji}, {dDeltaF_ji}\n")
f.close()
