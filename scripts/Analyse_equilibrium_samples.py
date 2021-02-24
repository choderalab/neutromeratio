import torch
import sys
import torch
from neutromeratio.parameter_gradients import setup_FEC
from neutromeratio.constants import (
    _get_names,
)
from neutromeratio.ani import AlchemicalANI1ccx, CompartimentedAlchemicalANI2x

#######################
#######################
# job idx
idx = int(sys.argv[1])
# where to write the results
base_path = str(sys.argv[2])
env = str(sys.argv[3])
potential_name = str(sys.argv[4])
max_snapshots_per_window = int(sys.argv[5])
if env == "droplet":
    diameter = int(sys.argv[6])
#######################
#######################

# name of the system
names = _get_names()
torch.set_num_threads(4)

name = names[idx - 1]
print(f"Analysing samples for tautomer pair: {name}")
print(f"Saving results in: {base_path}")
print(f"Using potential: {potential_name}")
print(f"Using {max_snapshots_per_window} snapshots/lambda")

if potential_name == "ANI1ccx":
    AlchemicalANI = AlchemicalANI1ccx
elif potential_name == "ANI2x":
    AlchemicalANI = CompartimentedAlchemicalANI2x
else:
    raise RuntimeError("Potential needs to be either ANI1ccx or ANI2x")


if env == "droplet":
    print("Simulating in environment: droplet.")
    fec = setup_FEC(
        name=name,
        data_path=base_path,
        ANImodel=AlchemicalANI,
        bulk_energy_calculation=True,
        env="droplet",
        max_snapshots_per_window=max_snapshots_per_window,
        diameter=diameter,
        load_pickled_FEC=False,
        include_restraint_energy_contribution=False,
    )
elif env == "vacuum":
    print("Simulating in environment: vacuum.")
    fec = setup_FEC(
        name=name,
        data_path=base_path,
        ANImodel=AlchemicalANI,
        bulk_energy_calculation=True,
        env="vacuum",
        max_snapshots_per_window=max_snapshots_per_window,
        load_pickled_FEC=False,
        include_restraint_energy_contribution=False,
    )
else:
    raise RuntimeError("No env specified. Aborting.")

DeltaF_ji, dDeltaF_ji = fec._end_state_free_energy_difference
if fec.flipped:
    DeltaF_ji *= -1
print(DeltaF_ji)

f = open(
    f"{base_path}/{potential_name}_{env}_rfe_results_in_kT_{max_snapshots_per_window}_snapshots.csv",
    "a+",
)
f.write(f"{name}, {DeltaF_ji}, {dDeltaF_ji}\n")
f.close()
