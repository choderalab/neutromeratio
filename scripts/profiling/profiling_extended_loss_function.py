from neutromeratio.parameter_gradients import (
    setup_and_perform_parameter_retraining_with_test_set_split,
)
from neutromeratio.ani import CompartimentedAlchemicalANI2x
import torch.autograd.profiler as profiler


# without pickled tautomer object
names = ["molDWRow_298", "SAMPLmol2", "SAMPLmol4"]
max_epochs = 100
model, model_name = CompartimentedAlchemicalANI2x, "CompartimentedAlchemicalANI2x"


(rmse_val, rmse_test,) = setup_and_perform_parameter_retraining_with_test_set_split(
    env="vacuum",
    checkpoint_filename=f"{model_name}_vacuum.pt",
    names=names,
    ANImodel=model,
    batch_size=3,
    data_path="../../data/test_data/vacuum",
    max_snapshots_per_window=300,
    bulk_energy_calculation=True,
    max_epochs=max_epochs,
    load_checkpoint=False,
    load_pickled_FEC=True,
    lr_AdamW=1e-5,
    lr_SGD=1e-5,
    include_snapshot_penalty=True,
)

print(rmse_val)

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
