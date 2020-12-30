import neutromeratio
import sys
import torch

assert len(sys.argv) == 10
env = sys.argv[1]
elements = sys.argv[2]
data_path = sys.argv[3]
model_name = str(sys.argv[4])
max_snapshots_per_window = int(sys.argv[5])
batch_size = int(sys.argv[6])
test_size = float(sys.argv[7])
validation_size = float(sys.argv[8])
diameter = int(sys.argv[9])

print(f"Max nr of snapshots: {max_snapshots_per_window}")
print(f"Test set size: {test_size}")
print(f"Valiation set size: {validation_size}")

assert validation_size >= 0.1 and validation_size <= 0.9
assert test_size >= 0.1 and test_size <= 0.9
assert batch_size >= 1
assert max_snapshots_per_window >= 10

if model_name == "ANI2x":
    model = neutromeratio.ani.CompartimentedAlchemicalANI2x
    # model = neutromeratio.ani.AlchemicalANI2x
    print(f"Using {model_name}.")
elif model_name == "ANI1ccx":
    model = neutromeratio.ani.AlchemicalANI1ccx
    print(f"Using {model_name}.")
elif model_name == "ANI1x":
    model = neutromeratio.ani.AlchemicalANI1x
    print(f"Using {model_name}.")
else:
    raise RuntimeError(f"Unknown model name: {model_name}")


if env == "droplet":
    bulk_energy_calculation = False
    torch.set_num_threads(4)
    print(f"Diameter: {diameter}")
else:
    torch.set_num_threads(4)
    bulk_energy_calculation = True

max_epochs = 100

(
    rmse_validation,
    rmse_test,
) = neutromeratio.parameter_gradients.setup_and_perform_parameter_retraining_with_test_set_split(
    env=env,
    ANImodel=model,
    batch_size=batch_size,
    load_checkpoint=False,
    max_snapshots_per_window=max_snapshots_per_window,
    checkpoint_filename=f"parameters_{model_name}_{env}.pt",
    data_path=data_path,
    bulk_energy_calculation=bulk_energy_calculation,
    elements=elements,
    max_epochs=max_epochs,
    diameter=diameter,
    load_pickled_FEC=True,
    lr_AdamW=1e-4,
    lr_SGD=1e-4,
    weight_decay=1e-06,
    test_size=test_size,
    validation_size=validation_size,
)
