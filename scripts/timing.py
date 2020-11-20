from neutromeratio.tautomers import Tautomer
from neutromeratio.analysis import setup_alchemical_system_and_energy_function
import numpy as np
from neutromeratio.ani import AlchemicalANI2x, CompartimentedAlchemicalANI2x
from neutromeratio.utils import _get_traj
from simtk import unit
from openmmtools.utils import is_quantity_close
import torch.autograd.profiler as profiler

env = "droplet"
# env = "vacuum"
# system name
name = "molDWRow_298"

# read in pregenerated traj
traj_path = (
    f"../data/test_data/{env}/molDWRow_298/molDWRow_298_lambda_0.0000_in_{env}.dcd"
)
top_path = f"../data/test_data/{env}/molDWRow_298/molDWRow_298_in_droplet.pdb"

# test energies with neutromeratio AlchemicalANI objects
# with ANI2x
traj, top = _get_traj(traj_path, top_path, None)
coordinates = [x.xyz[0] for x in traj[:800]] * unit.nanometer

energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
    name=name,
    ANImodel=AlchemicalANI2x,
    env=env,
    diameter=10,
    base_path=f"../data/test_data/{env}/{name}",
)

with profiler.profile(record_shapes=True, profile_memory=True) as prof:
    with profiler.record_function("model_inference"):
        AlchemicalANI2x_energy_lambda_1 = energy_function.calculate_energy(
            coordinates, lambda_value=1.0, include_restraint_energy_contribution=False
        )
s = prof.self_cpu_time_total / 1000000
print(f"time to compute energies in batch: {s:.3f} s")

energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
    name=name,
    ANImodel=CompartimentedAlchemicalANI2x,
    env=env,
    diameter=10,
    base_path=f"../data/test_data/{env}/{name}",
)

with profiler.profile(record_shapes=True, profile_memory=True) as prof:
    with profiler.record_function("model_inference"):
        CompartimentedAlchemicalANI2x_1 = energy_function.calculate_energy(
            coordinates, lambda_value=1.0, include_restraint_energy_contribution=False
        )
s = prof.self_cpu_time_total / 1000000
print(f"First time to calculate energies with CompartimentedAlchemicalANI2x: {s:.3f} s")

with profiler.profile(record_shapes=True, profile_memory=True) as prof:
    with profiler.record_function("model_inference"):
        CompartimentedAlchemicalANI2x_1 = energy_function.calculate_energy(
            coordinates, lambda_value=1.0, include_restraint_energy_contribution=False
        )
s = prof.self_cpu_time_total / 1000000
print(
    f"Second time to calculate energies with CompartimentedAlchemicalANI2x: {s:.3f} s"
)


for e1, e2 in zip(
    AlchemicalANI2x_energy_lambda_1.energy, CompartimentedAlchemicalANI2x_1.energy
):
    assert is_quantity_close(
        e1.in_units_of(unit.kilojoule_per_mole),
        e2.in_units_of(unit.kilojoule_per_mole),
        rtol=1e-5,
    )
