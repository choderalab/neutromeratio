import neutromeratio
from neutromeratio.constants import kT
from simtk import unit
import numpy as np
import pickle
import mdtraj as md
import torchani
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import random, sys
from neutromeratio.constants import platform, device, exclude_set
from neutromeratio import qm
from psi4.driver.p4util.exceptions import OptimizationConvergenceError

# job idx
idx = int(sys.argv[1])
mode = str(sys.argv[2])
base='/home/mwieder/Work/Projects/neutromeratio/data/results/csv'

try:
    assert(mode == 'ANI1x' or mode == 'ANI1ccx')
except AssertionError:
    raise AssertionError('Only ANI1x/ANI1ccx are accepted.')
# read in exp results, smiles and names
exp_results = pickle.load(open('/home/mwieder/Work/Projects/neutromeratio/data/exp_results.pickle', 'rb'))

# name of the system
protocoll = []
for name in sorted(exp_results):
    if name in exclude_set:
        continue
    protocoll.append(name)

name = protocoll[idx-1]
print(name)

# extract smiles
t1_smiles = exp_results[name]['t1-smiles']
t2_smiles = exp_results[name]['t2-smiles']

t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name,
                                                                t1_smiles,
                                                                t2_smiles,
                                                                nr_of_conformations=100,
                                                                enforceChirality=False)
tautomer = tautomers[0]
tautomer.perform_tautomer_transformation()

# set model
if mode == 'ANI1x': model = neutromeratio.ani.PureANI1x() 
if mode == 'ANI1ccx': model = neutromeratio.ani.PureANI1ccx()  

model = model.to(device)
torch.set_num_threads(1)

# t1
t1_e_ani = []
t1_e_qm = []
# calculate energy using both structures and pure ANI1ccx
energy_function = neutromeratio.ANI1_force_and_energy(
                                        model = model,
                                        mol = None,
                                        atoms = tautomer.initial_state_ligand_atoms)

  
mol = tautomer.initial_state_mol
for conf_idx in range(mol.GetNumConformers()):
    print(f"Starting with optimizing conf: {conf_idx}")
    try:
        psi4_mol = qm.mol2psi4(mol, conf_idx)
        e_qm, wfn = qm.optimize(psi4_mol)
        print(e_qm)
    except OptimizationConvergenceError as ex:
        print(f"Optimization failed for conf: {conf_idx}")
        continue
    
    conf = np.asarray(wfn.molecule().geometry()) * unit.bohr #!!!!!!!!!!! BOHR!
    e_ani, _, __, ____ = energy_function.calculate_energy(conf)
    t1_e_ani.append(e_ani/kT)
    t1_e_qm.append(e_qm/kT)
    break


# t2
t2_e_ani = []
t2_e_qm = []
# calculate energy using both structures and pure ANI1ccx
energy_function = neutromeratio.ANI1_force_and_energy(
                                        model = model,
                                        mol = None,
                                        atoms = tautomer.final_state_ligand_atoms)

  
mol = tautomer.final_state_mol
for conf_idx in range(mol.GetNumConformers()):
    print(f"Starting with optimizing conf: {conf_idx}")
    try:
        psi4_mol = qm.mol2psi4(mol, conf_idx)
        e_qm, wfn = qm.optimize(psi4_mol)
        print(e_qm)
    except OptimizationConvergenceError as ex:
        print(f"Optimization failed for conf: {conf_idx}")
        continue
    
    conf = np.asarray(psi4_mol.geometry()) * unit.bohr #!!! another distance unit was really missing  
    e_ani, _, __, ____ = energy_function.calculate_energy(conf)
    t2_e_ani.append(e_ani/kT)
    t2_e_qm.append(e_qm/kT)
    break

# write dE
f = open(f"{base}/{name}_t1.csv", 'w')
for conf_id, (e_ani, e_qm) in enumerate(zip(t1_e_ani, t1_e_qm)):
    f.write(f"{conf_id}, {e_ani}, {e_qm}\n")  
f.close()
# write dG
f = open(f"{base}/{name}_t2.csv", 'w')
for conf_id, (e_ani, e_qm) in enumerate(zip(t2_e_ani, t2_e_qm)):
    f.write(f"{conf_id}, {e_ani}, {e_qm}\n")  
f.close()
