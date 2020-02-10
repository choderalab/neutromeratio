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
from neutromeratio.constants import device, platform, kT, exclude_set
import sys
import os


# name of the system
idx = int(sys.argv[1])
# number of steps
n_steps = int(sys.argv[2])
# where to write the results
base_path = str(sys.argv[3])
# env
env = str(sys.argv[4])
assert(env == 'droplet' or env == 'vacuum')
# diameter
if env == 'droplet':
    diameter_in_angstrom = int(sys.argv[5])

protocol = []
exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))
for name in sorted(exp_results):
    if name in exclude_set:
        continue
    for lamb in np.linspace(0, 1, 11):
        protocol.append((name, np.round(lamb, 2)))

name, lambda_value = protocol[idx-1]
print(name)
print(lambda_value)


t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']

os.makedirs(f"{base_path}/{name}", exist_ok=True)

t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles)
for kappa_value, tautomer in enumerate(tautomers):
    kappa_value = float(kappa_value)
    print(kappa_value)
    tautomer.perform_tautomer_transformation()

    if env == 'droplet':
        m = tautomer.add_droplet(tautomer.hybrid_topology, 
                            tautomer.hybrid_coords, 
                            diameter=diameter_in_angstrom * unit.angstrom,
                            restrain_hydrogen_bonds=True,
                            restrain_hydrogen_angles=False,
                            top_file=f"{base_path}/{name}/{name}_kappa_{round(kappa_value)}_in_droplet.pdb")
    else:
        pdb_filepath = f"{base_path}/{name}/{name}_{round(kappa_value)}.pdb"
        try:
            traj = md.load(pdb_filepath)
        except OSError:
            coordinates = tautomer.hybrid_coords
            traj = md.Trajectory(coordinates.value_in_unit(unit.nanometer), tautomer.hybrid_topology)
            traj.save_pdb(pdb_filepath)
        # set coordinates #NOTE: note the xyz[0]
        tautomer.hybrid_coords = traj.xyz[0] * unit.nanometer

    # define the alchemical atoms
    alchemical_atoms = [tautomer.hybrid_hydrogen_idx_at_lambda_1, tautomer.hybrid_hydrogen_idx_at_lambda_0]

    # extract hydrogen donor idx and hydrogen idx for from_mol
    model = neutromeratio.ani.LinearAlchemicalSingleTopologyANI(alchemical_atoms=alchemical_atoms)
    model = model.to(device)
    torch.set_num_threads(1)

    # perform initial sampling
    if env == 'vacuum':
        energy_function = neutromeratio.ANI1_force_and_energy(
            model=model,
            atoms=tautomer.hybrid_atoms,
            mol=None,
        )
    else:
        energy_function = neutromeratio.ANI1_force_and_energy(
            model=model,
            atoms=tautomer.ligand_in_water_atoms,
            mol=None,
        )


    for r in tautomer.ligand_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)

    for r in tautomer.hybrid_ligand_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)

    if env == 'droplet':
        tautomer.add_COM_for_hybrid_ligand(np.array([diameter_in_angstrom/2, diameter_in_angstrom/2, diameter_in_angstrom/2]) * unit.angstrom)

        for r in tautomer.solvent_restraints:
            energy_function.add_restraint_to_lambda_protocol(r)

        for r in tautomer.com_restraints:
            energy_function.add_restraint_to_lambda_protocol(r)


    energy_and_force = lambda x : energy_function.calculate_force(x, lambda_value)
    if env == 'droplet':
        x0 = tautomer.ligand_in_water_coordinates
        langevin = neutromeratio.LangevinDynamics(atoms=tautomer.ligand_in_water_atoms,
                                            energy_and_force=energy_and_force)
    elif env == 'vacuum':
        x0 = tautomer.hybrid_coords
        langevin = neutromeratio.LangevinDynamics(atoms=tautomer.hybrid_atoms,
                                            energy_and_force=energy_and_force)
    else:
        raise RuntimeError()
    x0, e_history = energy_function.minimize(x0, maxiter=5000, lambda_value=lambda_value)

    equilibrium_samples, energies, restraint_bias, stddev, ensemble_bias = langevin.run_dynamics(x0,
                                                                        n_steps=n_steps,
                                                                        stepsize=1.0*unit.femtosecond,
                                                                        progress_bar=False)

    # save equilibrium energy values 
    for global_list, poperty_name in zip([energies, stddev, restraint_bias, ensemble_bias], ['energy', 'stddev', 'restraint_bias', 'ensemble_bias']):
        f = open(f"{base_path}/{name}/{name}_lambda_{lambda_value:0.4f}_{poperty_name}_kappa_{round(kappa_value)}_in_{env}.csv", 'w+')
        for e in global_list[::20]:
            e_unitless = e / kT
            f.write('{}\n'.format(e_unitless))
        f.close()   

    equilibrium_samples = [x.value_in_unit(unit.nanometer) for x in equilibrium_samples]
    if env == 'vacuum':
        ani_traj = md.Trajectory(equilibrium_samples[::20], tautomer.hybrid_topology)
    else:
        ani_traj = md.Trajectory(equilibrium_samples[::20], tautomer.ligand_in_water_topology)

    ani_traj.save(f"{base_path}/{name}/{name}_lambda_{lambda_value:0.4f}_kappa_{round(kappa_value)}_in_{env}.dcd", force_overwrite=True)
