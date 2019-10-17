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
from neutromeratio.constants import device, platform

exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))

# specify the system you want to simulate
name = 'molDWRow_298'
#name = 'molDWRow_37'
#name = 'molDWRow_45'
#name = 'molDWRow_160'
#name = 'molDWRow_590'


# number of steps
n_steps = 1000

exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))

t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']

# generate both rdkit mol
tautomer = neutromeratio.Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles), nr_of_conformations=20)
tautomer.perform_tautomer_transformation_forward()
tautomer.add_droplet(tautomer.hybrid_topology, tautomer.hybrid_coords)

# define the alchemical atoms
alchemical_atoms=[tautomer.hybrid_dummy_hydrogen, tautomer.hydrogen_idx]

np.random.seed(0)

# extract hydrogen donor idx and hydrogen idx for from_mol
model = neutromeratio.ani.LinearAlchemicalDualTopologyANI(alchemical_atoms=alchemical_atoms)
model = model.to(device)
torch.set_num_threads(2)

# perform initial sampling
energy_function = neutromeratio.ANI1_force_and_energy(
                                        model = model,
                                        atoms = tautomer.ligand_in_water_atoms,
                                        mol = tautomer.ligand_in_water_ase_mol,
                                        )

for r in tautomer.ligand_restraints:
    energy_function.add_restraint(r)

for r in tautomer.hybrid_ligand_restraints:
    energy_function.add_restraint(r)


trajectory = []
bias_list = []
for lambda_value in np.linspace(0,1,10):
    print(lambda_value)
    energy_and_force = lambda x : energy_function.calculate_force(x, lambda_value)
    langevin = neutromeratio.LangevinDynamics(atoms = tautomer.ligand_in_water_atoms,
                                temperature = 300*unit.kelvin,
                                energy_and_force = energy_and_force)
    x0 = np.array(tautomer.ligand_in_water_coordinates) * unit.angstrom
    x0 = energy_function.minimize(x0)

    
    equilibrium_samples, energies, bias = langevin.run_dynamics(x0, n_steps=n_steps, stepsize=0.5 * unit.femtosecond, progress_bar=True)
    bias_list.append(bias)
    trajectory.append(equilibrium_samples)
    
