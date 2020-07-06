import neutromeratio
from simtk import unit
import numpy as np
import pickle
import sys
import torch
from neutromeratio.parameter_gradients import FreeEnergyCalculator, setup_mbar
from neutromeratio.constants import kT, device, exclude_set_ANI, mols_with_charge, _get_names
from glob import glob
from neutromeratio.ani import AlchemicalANI1ccx

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
names = _get_names()

name = names[idx-1]
print(name)
fec = setup_mbar(name=name, 
                data_path=base_path,
                ANImodel=AlchemicalANI1ccx,
                bulk_energy_calculation=False,
                env='droplet',
                max_snapshots_per_window=200,
                diameter=18)

DeltaF_ji, dDeltaF_ji = fec.end_state_free_energy_difference
if fec.flipped:
    DeltaF_ji *= -1
print(DeltaF_ji)

f = open(f"{base_path}/AlchemicalANI1ccx_droplet_rfe_results_in_kT_200snapshots.csv", 'a+')
f.write(f"{name}, {DeltaF_ji}, {dDeltaF_ji}\n")
f.close()
