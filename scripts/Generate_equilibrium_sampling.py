from simtk import unit
import numpy as np
from tqdm import tqdm
import mdtraj as md
import nglview
from rdkit import Chem
from rdkit.Chem import AllChem
import neutromeratio
import matplotlib.pyplot as plt
import pickle
import torchani
import torch
from neutromeratio.constants import _get_names, device, platform, kT, exclude_set_ANI, mols_with_charge, multiple_stereobonds
import sys
import os
from neutromeratio.analysis import setup_alchemical_system_and_energy_function
from neutromeratio.ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x

# name of the system
idx = int(sys.argv[1])
# number of steps
n_steps = int(sys.argv[2])
# where to write the results
base_path = str(sys.argv[3])
# env
env = str(sys.argv[4])

if not (env == 'droplet' or env == 'vacuum'):
    raise RuntimeError('Env must be `droplet` or `vacuum`. Aborting.')
# diameter
if env == 'droplet':
    diameter_in_angstrom = int(sys.argv[5])
    if not diameter_in_angstrom or diameter_in_angstrom < 1:
        raise RuntimeError('Diameter must be above 1 Angstrom')

protocol = []
names = neutromeratio.parameter_gradients._get_names()
for name in names:
    for lamb in np.linspace(0, 1, 11):
        protocol.append((name, np.round(lamb, 2)))

name, lambda_value = protocol[idx-1]
print(name)
print(lambda_value)

energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
    name=name,
    ANImodel=AlchemicalANI1ccx,
    env=env,
    diameter=diameter_in_angstrom
    base_path=base_path)

energy_and_force = lambda x: energy_function.calculate_force(x, lambda_value)

if env == 'droplet':
    x0 = tautomer.get_ligand_in_water_coordinates()
    langevin = neutromeratio.LangevinDynamics(atoms=tautomer.ligand_in_water_atoms,
                                        energy_and_force=energy_and_force)
elif env == 'vacuum':
    x0 = tautomer.get_hybrid_coordinates()
    langevin = neutromeratio.LangevinDynamics(atoms=tautomer.hybrid_atoms,
                                        energy_and_force=energy_and_force)
else:
    raise RuntimeError()

x0, e_history = energy_function.minimize(x0, maxiter=5000, lambda_value=lambda_value)
equilibrium_samples, energies, restraint_contribution = langevin.run_dynamics(x0,
                                                                    n_steps=n_steps,
                                                                    stepsize=0.5*unit.femtosecond,
                                                                    progress_bar=False)

# save equilibrium energy values 
for global_list, poperty_name in zip([energies, restraint_contribution], ['energy', 'restraint_energy_contribution']):
    f = open(f"{base_path}/{name}/{name}_lambda_{lambda_value:0.4f}_{poperty_name}_in_{env}.csv", 'w+')
    for e in global_list[::20]:
        e_unitless = e / kT
        f.write('{}\n'.format(e_unitless))
    f.close()   

equilibrium_samples = [x.value_in_unit(unit.nanometer) for x in equilibrium_samples]
if env == 'vacuum':
    ani_traj = md.Trajectory(equilibrium_samples[::20], tautomer.hybrid_topology)
else:
    ani_traj = md.Trajectory(equilibrium_samples[::20], tautomer.ligand_in_water_topology)

ani_traj.save(f"{base_path}/{name}/{name}_lambda_{lambda_value:0.4f}_in_{env}.dcd", force_overwrite=True)
