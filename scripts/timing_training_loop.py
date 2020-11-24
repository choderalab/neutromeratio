import torch.autograd.profiler as profiler
from neutromeratio.parameter_gradients import (
    setup_and_perform_parameter_retraining_with_test_set_split,
)
import os
from neutromeratio.ani import CompartimentedAlchemicalANI2x, AlchemicalANI2x
import numpy as np


def _remove_files(name, max_epochs=1):
    try:
        os.remove(f"{name}.pt")
    except FileNotFoundError:
        pass
    for i in range(1, max_epochs):
        os.remove(f"{name}_{i}.pt")
    os.remove(f"{name}_best.pt")


names = ["molDWRow_298"]
env = "droplet"
diameter = 10
max_snapshots_per_window = 10
max_epochs = 3

model = CompartimentedAlchemicalANI2x
model_name = "CompartimentedAlchemicalANI2x"
# model._reset_parameters()

with profiler.profile(record_shapes=True, profile_memory=True) as prof:
    with profiler.record_function("model_inference"):

        (
            rmse_val,
            rmse_test,
        ) = setup_and_perform_parameter_retraining_with_test_set_split(
            env=env,
            names=names,
            ANImodel=model,
            batch_size=1,
            max_snapshots_per_window=max_snapshots_per_window,
            checkpoint_filename=f"{model_name}_droplet.pt",
            data_path=f"../data/test_data/{env}",
            nr_of_nn=8,
            max_epochs=max_epochs,
            diameter=diameter,
            load_checkpoint=False,
            load_pickled_FEC=True,
        )

s = prof.self_cpu_time_total / 1000000
print(f"time to compute 3 epochs: {s:.3f} s")


try:
    assert np.isclose(rmse_val[-1], rmse_test)
    assert np.isclose(rmse_val[0], 16.44867706298828)
    assert np.isclose(rmse_val[-1], 3.080655097961426, rtol=1e-4)
finally:
    _remove_files(model_name + "_droplet", max_epochs)
    print(rmse_val, rmse_test)
