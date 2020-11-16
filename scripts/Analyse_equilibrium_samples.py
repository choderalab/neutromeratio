import neutromeratio
from simtk import unit
import numpy as np
import pickle
import sys
import torch
from neutromeratio.parameter_gradients import setup_FEC
from neutromeratio.constants import (
    kT,
    device,
    exclude_set_ANI,
    mols_with_charge,
    _get_names,
)
from glob import glob
from neutromeratio.ani import AlchemicalANI1ccx, AlchemicalANI2x

#######################
#######################
# job idx
idx = int(sys.argv[1])
# where to write the results
base_path = str(sys.argv[2])
env = str(sys.argv[3])
potential_name = str(sys.argv[4])
#######################
#######################
# read in exp results, smiles and names
exp_results = pickle.load(open("../data/exp_results.pickle", "rb"))

# name of the system
names = _get_names()
torch.set_num_threads(4)

name = names[idx - 1]
print(f"Analysing samples for tautomer pair: {name}")
print(f"Saving results in: {base_path}")
print(f"Using potential: {potential_name}")

if potential_name == "ANI1ccx":
    AlchemicalANI = AlchemicalANI1ccx
elif potential_name == "ANI2x":
    AlchemicalANI = AlchemicalANI2x
else:
    raise RuntimeError("Potential needs to be either ANI1ccx or ANI2x")


if env == "droplet":
    print("Simulating in environment: droplet.")
    fec = setup_FEC(
        name=name,
        data_path=base_path,
        ANImodel=AlchemicalANI,
        bulk_energy_calculation=False,
        env="droplet",
        max_snapshots_per_window=300,
        diameter=18,
    )
elif env == "vacuum":
    print("Simulating in environment: vacuum.")
    fec = setup_FEC(
        name=name,
        data_path=base_path,
        ANImodel=AlchemicalANI,
        bulk_energy_calculation=True,
        env="vacuum",
        max_snapshots_per_window=300,
    )
else:
    raise RuntimeError("No env specified. Aborting.")

DeltaF_ji, dDeltaF_ji = fec._end_state_free_energy_difference
if fec.flipped:
    DeltaF_ji *= -1
print(DeltaF_ji)

f = open(f"{base_path}/{potential_name}_{env}_rfe_results_in_kT_300snapshots.csv", "a+")
f.write(f"{name}, {DeltaF_ji}, {dDeltaF_ji}\n")
f.close()
