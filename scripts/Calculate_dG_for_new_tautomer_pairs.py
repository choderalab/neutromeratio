import neutromeratio
from simtk import unit
import numpy as np
import pickle
import sys
import torch
from neutromeratio.parameter_gradients import setup_mbar_for_new_tautomer_pairs
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
smiles1 = str(sys.argv[1])
smiles2 = str(sys.argv[2])
name = str(sys.argv[3])
base_path = str(sys.argv[4])
env = str(sys.argv[5])
potential_name = str(sys.argv[6])
#######################
#######################
torch.set_num_threads(4)

print(f"Analysing samples for tautomer pair: {name}")
print(f"With SMILES1: {smiles1}")
print(f"With SMILES2: {smiles2}")
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
    fec = setup_mbar_for_new_tautomer_pairs(
        name=name,
        t1_smiles=smiles1,
        t2_smiles=smiles2,
        data_path=base_path,
        ANImodel=AlchemicalANI,
        bulk_energy_calculation=False,
        env="droplet",
        max_snapshots_per_window=300,
        diameter=-1,
    )
elif env == "vacuum":
    print("Simulating in environment: vacuum.")
    fec = setup_mbar_for_new_tautomer_pairs(
        name=name,
        t1_smiles=smiles1,
        t2_smiles=smiles2,
        data_path=base_path,
        ANImodel=AlchemicalANI,
        bulk_energy_calculation=True,
        env="vacuum",
        max_snapshots_per_window=300,
        checkpoint_file="../data/retraining/parameters_ANI1ccx_vacuum_best.pt",
    )
else:
    raise RuntimeError("No env specified. Aborting.")

DeltaF_ji, dDeltaF_ji = fec._end_state_free_energy_difference
print(f"dG obtained from vacuum simulations: {DeltaF_ji} with {dDeltaF_ji} [kT]")

if potential_name == "ANI1ccx":
    DeltaF_ji = fec._compute_free_energy_difference()
    print(
        f"dG obtained with optimized parameters from vacuum simulations: {DeltaF_ji} [kT]"
    )

else:
    print("We only provide optimized parameters for ANI1ccx.")
