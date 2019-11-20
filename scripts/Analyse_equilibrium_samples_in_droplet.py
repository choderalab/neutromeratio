import neutromeratio
from openmmtools.constants import kB
from simtk import unit
import numpy as np
import pickle
import mdtraj as md
import matplotlib.pyplot as plt
import sys
import torch
from neutromeratio.parameter_gradients import FreeEnergyCalculator
from neutromeratio.constants import kT, device, exclude_set
from glob import glob

def parse_lambda_from_dcd_filename(dcd_filename, env):
    return float(dcd_filename[:dcd_filename.find(f"_in_{env}")].split('_')[-1])

# job idx
idx = int(sys.argv[1])
# where to write the results
base_path = str(sys.argv[2])
env = str(sys.argv[3])
assert(env == 'droplet' or env == 'vacuum')
# diameter
if env == 'droplet':
   diameter_in_angstrom = int(sys.argv[4])
#######################

mode = 'forward'

# read in exp results, smiles and names
exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))

# name of the system
protocoll = []
for name in sorted(exp_results):
    if name in exclude_set:
        continue
    protocoll.append(name)

name = protocoll[idx-1]
print(name)

# don't change - direction is fixed for all runs
#################
t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']


# generate both rdkit mol
tautomer = neutromeratio.Tautomer(name=name, initial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles), nr_of_conformations=20)
if mode == 'forward':
    tautomer.perform_tautomer_transformation_forward()
elif mode == 'reverse':
    tautomer.perform_tautomer_transformation_reverse()
else:
    raise RuntimeError('No tautomer reaction direction was specified.')

if env == 'droplet':
    tautomer.add_droplet(tautomer.hybrid_topology, 
                                tautomer.hybrid_coords, 
                                diameter=diameter_in_angstrom * unit.angstrom,
                                restrain_hydrogens=True,
                                file=f"{base_path}/{name}/{name}_in_droplet_{mode}.pdb")
    print('Nr of atoms: {}'.format(len(tautomer.ligand_in_water_atoms)))
    atoms = tautomer.ligand_in_water_atoms
    top = tautomer.ligand_in_water_topology
else:
    atoms = tautomer.hybrid_atoms
    top = tautomer.hybrid_topology

# define the alchemical atoms
alchemical_atoms=[tautomer.hybrid_hydrogen_idx_at_lambda_1, tautomer.hybrid_hydrogen_idx_at_lambda_0]


# extract hydrogen donor idx and hydrogen idx for from_mol
model = neutromeratio.ani.LinearAlchemicalDualTopologyANI(alchemical_atoms=alchemical_atoms, adventure_mode=True)
model = model.to(device)
torch.set_num_threads(2)

# perform initial sampling
energy_function = neutromeratio.ANI1_force_and_energy(
                                        model = model,
                                        atoms = atoms,
                                        mol = None,
                                        )


for r in tautomer.ligand_restraints:
    energy_function.add_restraint(r)

for r in tautomer.hybrid_ligand_restraints:
    energy_function.add_restraint(r)

if env == 'droplet':

    tautomer.add_COM_for_hybrid_ligand(np.array([diameter_in_angstrom/2, diameter_in_angstrom/2, diameter_in_angstrom/2]) * unit.angstrom)

    for r in tautomer.solvent_restraints:
        energy_function.add_restraint(r)

    for r in tautomer.com_restraints:
        energy_function.add_restraint(r)


# get steps inclusive endpoints
# and lambda values in list
dcds = glob(f"{base_path}/{name}/*.dcd")

lambdas = []
ani_trajs = []
energies = []

for dcd_filename in dcds:
    lam = parse_lambda_from_dcd_filename(dcd_filename, env)
    lambdas.append(lam)
    traj = md.load_dcd(dcd_filename, top=top)
    print(len(traj))
    ani_trajs.append(traj)  
    f = open(f"{base_path}/{name}/{name}_lambda_{lam:0.4f}_energy_in_{env}_{mode}.csv", 'r')  
    tmp_e = []
    for e in f:
        tmp_e.append(float(e))
    f.close()
    energies.append(np.array(tmp_e))

# plotting the energies for all equilibrium runs
for e in energies: 
    plt.plot(e, alpha=0.5)
plt.show()
plt.savefig(f"{base_path}/{name}/{name}_energy.png")

# calculate free energy in kT
fec = FreeEnergyCalculator(ani_model=energy_function, 
                            ani_trajs=ani_trajs, 
                            potential_energy_trajs=energies, 
                            lambdas=lambdas,
                            nr_of_atoms=len(atoms),
                            max_snapshots_per_window=200)
DeltaF_ji, dDeltaF_ji = fec.end_state_free_energy_difference
print(fec.end_state_free_energy_difference)
f = open(f"{base_path}/energies.csv", 'a+')
f.write(f"{name}, {DeltaF_ji}, {dDeltaF_ji}\n")
f.close()
