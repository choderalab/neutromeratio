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
# envenv = str(sys.argv[3])
#######################
#######################
# read in exp results, smiles and names
exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))

# name of the system
names = _get_names()

name = names[idx-1]
print(name)
print(base_path)
if env == 'droplet':
    print('Env: droplet.')
    fec = setup_mbar(name=name, 
                    data_path=base_path,
                    ANImodel=AlchemicalANI1ccx,
                    bulk_energy_calculation=False,
                    env='droplet',
                    max_snapshots_per_window=500,
                    diameter=18)
elif env == 'vacuum':
    print('Env: vacuum.')
    fec = setup_mbar(name=name, 
                    data_path=base_path,
                    ANImodel=AlchemicalANI1ccx,
                    bulk_energy_calculation=False,
                    env='vacuum',
                    max_snapshots_per_window=500
    )
else:
    raise RuntimeError('No env specified. Aborting.')

DeltaF_ji, dDeltaF_ji = fec.end_state_free_energy_difference
if fec.flipped:
    DeltaF_ji *= -1
print(DeltaF_ji)

f = open(f"{base_path}/AlchemicalANI1ccx_{env}_rfe_results_in_kT_500snapshots.csv", 'a+')
f.write(f"{name}, {DeltaF_ji}, {dDeltaF_ji}\n")
f.close()
