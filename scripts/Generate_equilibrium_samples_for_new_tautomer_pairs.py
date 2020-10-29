from simtk import unit
import numpy as np
import mdtraj as md
import neutromeratio
from neutromeratio.constants import _get_names, device, platform, kT
import sys
import os
import torch
from neutromeratio.analysis import setup_new_alchemical_system_and_energy_function
from neutromeratio.ani import AlchemicalANI2x

smiles1 = str(sys.argv[1])
smiles2 = str(sys.argv[2])
name = str(sys.argv[3])
lambda_value = float(sys.argv[4])
n_steps = int(sys.argv[5])
base_path = str(sys.argv[6])  # where to write the results
env = str(sys.argv[7])
model_name = str(sys.argv[8])

print(f"Generating samples for tautomer pair: {name}")
print(f"With SMILES1: {smiles1}")
print(f"With SMILES2: {smiles2}")
print(f"Lambda coupling parameter: {lambda_value} (between 0. and 1.)")
print(f"Saving results in: {base_path}/{name}")
print(f"Using potential: {model_name}")
print(f"Simulating in environment: {env}")

if model_name == "ANI2x":
    model = neutromeratio.ani.AlchemicalANI2x
    print(f"Using {model_name}.")
elif model_name == "ANI1ccx":
    model = neutromeratio.ani.AlchemicalANI1ccx
    print(f"Using {model_name}.")
elif model_name == "ANI1x":
    model = neutromeratio.ani.AlchemicalANI1x
    print(f"Using {model_name}.")
else:
    raise RuntimeError(f"Unknown model name: {model_name}")


# INPUT CHECKS
if not (env in ("droplet", "vacuum")):
    raise RuntimeError("Env must be `droplet` or `vacuum`. Aborting.")
if env == "droplet":
    diameter_in_angstrom = int(sys.argv[6])
    if not diameter_in_angstrom or diameter_in_angstrom < 1:
        raise RuntimeError("Diameter must be above 1 Angstrom")
else:
    diameter_in_angstrom = -1

# Generating base path
base_path = f"{base_path}/{name}"
os.makedirs(base_path, exist_ok=True)

# setting up energy function
energy_function, tautomer = setup_new_alchemical_system_and_energy_function(
    name=name,
    t1_smiles=smiles1,
    t2_smiles=smiles2,
    env=env,
    ANImodel=model,
    diameter=diameter_in_angstrom,
    base_path=base_path,
)

energy_and_force = lambda x: energy_function.calculate_force(x, lambda_value)

if env == "droplet":
    x0 = tautomer.get_ligand_in_water_coordinates()
    langevin = neutromeratio.LangevinDynamics(
        atoms=tautomer.ligand_in_water_atoms, energy_and_force=energy_and_force
    )
elif env == "vacuum":
    x0 = tautomer.get_hybrid_coordinates()
    langevin = neutromeratio.LangevinDynamics(
        atoms=tautomer.hybrid_atoms, energy_and_force=energy_and_force
    )
else:
    raise RuntimeError()

# pytorch number of threads to be used
torch.set_num_threads(1)

# start minimization
x0, e_history = energy_function.minimize(x0, maxiter=5000, lambda_value=lambda_value)

# start sampling
equilibrium_samples, energies, restraint_contribution = langevin.run_dynamics(
    x0, n_steps=n_steps, stepsize=0.5 * unit.femtosecond, progress_bar=True
)

# save equilibrium energy values
for global_list, poperty_name in zip(
    [energies, restraint_contribution], ["energy", "restraint_energy_contribution"]
):
    f = open(
        f"{base_path}/{name}_lambda_{lambda_value:0.4f}_{poperty_name}_in_{env}.csv",
        "w+",
    )
    for e in global_list[::20]:
        e_unitless = e / kT
        f.write("{}\n".format(e_unitless))
    f.close()

# save trajectory
equilibrium_samples = [x[0].value_in_unit(unit.nanometer) for x in equilibrium_samples]
if env == "vacuum":
    ani_traj = md.Trajectory(equilibrium_samples[::20], tautomer.hybrid_topology)
else:
    ani_traj = md.Trajectory(
        equilibrium_samples[::20], tautomer.ligand_in_water_topology
    )

ani_traj.save(
    f"{base_path}/{name}_lambda_{lambda_value:0.4f}_in_{env}.dcd", force_overwrite=True
)
