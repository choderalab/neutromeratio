# This script calculates the free energy estimate given a checkpoint file
# and writes out the result in the training directory

import torch
import pickle
import torch
from neutromeratio.parameter_gradients import (
    setup_FEC,
    get_perturbed_free_energy_difference,
)
from neutromeratio.constants import (
    _get_names,
)
from neutromeratio.ani import CompartimentedAlchemicalANI2x, AlchemicalANI1ccx
import argparse

## hardcoding some params
torch.set_num_threads(4)
diameter = 16
########

### parse command line arg
parser = argparse.ArgumentParser()
parser.add_argument("idx", action="store", type=int)
parser.add_argument("base_path", action="store", type=str)
parser.add_argument("env", action="store", type=str)
parser.add_argument("potential_name", action="store", type=str)
parser.add_argument("snapshots", action="store", type=int)
parser.add_argument(
    "-c",
    "--checkpoint_file",
    action="store",
    required=True,
    type=str,
)
args = parser.parse_args()
max_snapshots_per_window = args.snapshots
# load test/training/validation set
all_names: dict = pickle.load(open("training_validation_tests.pickle", "br"))

if args.potential_name == "ANI1ccx":
    AlchemicalANI = AlchemicalANI1ccx
elif args.potential_name == "ANI2x":
    AlchemicalANI = CompartimentedAlchemicalANI2x
else:
    raise RuntimeError("Potential needs to be either ANI1ccx or ANI2x")

# initialize
_ = AlchemicalANI([0, 0])

# name of the system
names = _get_names()
name = names[args.idx - 1]
print(f"Analysing samples for tautomer pair: {name}")
print(f"Using potential: {AlchemicalANI.name}")

which_set_does_it_belong_to = ""
if all_names[name] == "training":
    which_set_does_it_belong_to = "training"
elif all_names[name] == "testing":
    which_set_does_it_belong_to = "testing"
elif all_names[name] == "validating":
    which_set_does_it_belong_to = "validating"
else:
    print("######################################")
    raise RuntimeError("That should not have happend.")

checkpoint_file_base = args.checkpoint_file.split(".")[0]

if args.env == "droplet":
    print("Simulating in environment: droplet.")
    fec = setup_FEC(
        name=name,
        data_path=args.base_path,
        ANImodel=AlchemicalANI,
        bulk_energy_calculation=True,
        env="droplet",
        max_snapshots_per_window=max_snapshots_per_window,
        diameter=diameter,
        load_pickled_FEC=True,
        save_pickled_FEC=False,
        include_restraint_energy_contribution=False,
    )
elif args.env == "vacuum":
    print("Simulating in environment: vacuum.")
    fec = setup_FEC(
        name=name,
        data_path=args.base_path,
        ANImodel=AlchemicalANI,
        bulk_energy_calculation=True,
        env="vacuum",
        max_snapshots_per_window=max_snapshots_per_window,
        load_pickled_FEC=False,
        save_pickled_FEC=False,
        include_restraint_energy_contribution=False,
    )
else:
    raise RuntimeError("No env specified. Aborting.")

# load perturbed ani parameters
fec.ani_model.model.load_nn_parameters(args.checkpoint_file)

# calculate perturbed free energy
DeltaF_ji = get_perturbed_free_energy_difference(fec).item()
# get MAE for snapshot deviation
snapshot_penalty = fec.mae_between_potentials_for_snapshots()

with open(
    f"{checkpoint_file_base}_rfe_results_in_kT_{max_snapshots_per_window}_snapshots.csv",
    "a+",
) as f:
    f.write(f"{name}, {which_set_does_it_belong_to}, {DeltaF_ji}, {snapshot_penalty}\n")
