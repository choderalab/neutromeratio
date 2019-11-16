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
from neutromeratio.constants import kT, device

path = 'neutromeratio_results_nov15/SAMPLmol2/'
from glob import glob

dcds = glob(path + '*.dcd')

# read in exp results, smiles and names
exp_results = pickle.load(open('/Users/joshuafass/Documents/Github/neutromeratio/data/exp_results.pickle', 'rb'))

exclude = ['molDWRow_1004', 'molDWRow_1110', 'molDWRow_1184', 'molDWRow_1185', 'molDWRow_1189', 'molDWRow_1262', 'molDWRow_1263',
'molDWRow_1267', 'molDWRow_1275', 'molDWRow_1279', 'molDWRow_1280', 'molDWRow_1282', 'molDWRow_1283', 'molDWRow_553',
'molDWRow_557', 'molDWRow_580', 'molDWRow_581', 'molDWRow_582', 'molDWRow_615', 'molDWRow_616', 'molDWRow_617',
'molDWRow_618', 'molDWRow_643', 'molDWRow_758', 'molDWRow_82', 'molDWRow_83', 'molDWRow_952', 'molDWRow_953',
'molDWRow_955', 'molDWRow_988', 'molDWRow_989', 'molDWRow_990', 'molDWRow_991', 'molDWRow_992']

# name of the system
protocoll = []
for name in sorted(exp_results):
    if name in exclude:
        continue
    protocoll.append(name)
idx = 1
name = protocoll[idx-1]
print(name)



def prepare(name, mode="forward"):

    diameter_in_angstrom = 18
    base_path = path

    # don't change - direction is fixed for all runs
    #################
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']


    # generate both rdkit mol
    tautomer = neutromeratio.Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles), nr_of_conformations=20)
    if mode == 'forward':
        tautomer.perform_tautomer_transformation_forward()
    elif mode == 'reverse':
        tautomer.perform_tautomer_transformation_reverse()
    else:
        raise RuntimeError('No tautomer reaction direction was specified.')

    tautomer.add_droplet(tautomer.hybrid_topology,
                                tautomer.hybrid_coords,
                                diameter=diameter_in_angstrom * unit.angstrom,
                                restrain_hydrogens=True,
                                file=f"{base_path}/{name}_in_droplet_{mode}.pdb")

    # define the alchemical atoms
    alchemical_atoms=[tautomer.hybrid_dummy_hydrogen, tautomer.hydrogen_idx]

    print('Nr of atoms: {}'.format(len(tautomer.ligand_in_water_atoms)))

    # extract hydrogen donor idx and hydrogen idx for from_mol
    model = neutromeratio.ani.LinearAlchemicalDualTopologyANI(alchemical_atoms=alchemical_atoms)
    model = model.to(device)
    torch.set_num_threads(2)

    # perform initial sampling
    energy_function = neutromeratio.ANI1_force_and_energy(
                                            model = model,
                                            atoms = tautomer.ligand_in_water_atoms,
                                            mol = None,
                                            )

    tautomer.add_COM_for_hybrid_ligand(np.array([diameter_in_angstrom/2, diameter_in_angstrom/2, diameter_in_angstrom/2]) * unit.angstrom)

    for r in tautomer.ligand_restraints:
        energy_function.add_restraint(r)

    for r in tautomer.hybrid_ligand_restraints:
        energy_function.add_restraint(r)

    for r in tautomer.solvent_restraints:
        energy_function.add_restraint(r)

    for r in tautomer.com_restraints:
        energy_function.add_restraint(r)
    return model, tautomer, energy_function
model, tautomer, energy_function = prepare(name)

def parse_lambda_from_filename(filename):
    return float(filename[:filename.find('_in_droplet')].split('_')[-1])


from tqdm import tqdm


def compute_atomic_energy_contributions_without_dummy_atom(traj, model, energy_function, dummy_atom):
    n_models = energy_function.model.ensemble_size

    xyz_in_angstroms = traj.xyz * 10
    coordinates = torch.tensor(xyz_in_angstroms)
    n_snapshots = len(coordinates)

    species = energy_function.species
    mod_species = torch.cat((species[:, :dummy_atom], species[:, dummy_atom + 1:]), dim=1)

    species_list = list(map(int, mod_species[0]))
    n_atoms = len(species_list)

    print('n_snapshots: {}, n_atoms: {}, n_models: {}'.format(n_snapshots, n_atoms, n_models))

    mod_coordinates = torch.cat((coordinates[:, :dummy_atom], coordinates[:, dummy_atom + 1:]), dim=1)
    mod_species = torch.stack([mod_species[0]] * len(mod_coordinates))
    print('mod_species.shape', mod_species.shape)
    print('mod_coordinates.shape', mod_coordinates.shape)
    _, mod_aevs = model.aev_computer((mod_species, mod_coordinates))

    raw_energies = np.zeros((n_snapshots, n_atoms, n_models))
    for i in tqdm(range(n_atoms)):
        for j in range(n_models):
            raw_energies[:, i, j] = model.neural_networks[j][species_list[i]].forward(
                mod_aevs[:, i, :]).detach().flatten()
    return raw_energies


def compute_endstate_atomic_energy_contributions(traj, model, energy_function):
    assert (len(model.alchemical_atoms) == 2)

    if len(traj) > 300: print('warning! may want to break traj into bite-size chunks')

    dummy_0, dummy_1 = model.alchemical_atoms
    print('(dummy_0, dummy_1)', dummy_0, dummy_1)

    raw_energies_without_dummy_0 = compute_atomic_energy_contributions_without_dummy_atom(
        traj, model, energy_function, dummy_0)

    raw_energies_without_dummy_1 = compute_atomic_energy_contributions_without_dummy_atom(
        traj, model, energy_function, dummy_1)

    return raw_energies_without_dummy_0, raw_energies_without_dummy_1

from neutromeratio.ani import hartree_to_kJ_mol


def compute_total_uncertainties(raw_energies):
    total_uncertainties = (raw_energies * hartree_to_kJ_mol).sum(1).std(-1)
    return total_uncertainties


def compute_linear_penalty(raw_energies_0, raw_energies_1,
                           lambda_value, per_atom_thresh=0.5):
    """raw_energies in hartree, per_atom_thresh in kj/mol"""
    n_atoms = raw_energies_0.shape[1]
    total_thresh = per_atom_thresh * n_atoms

    total_uncertainties_0 = compute_total_uncertainties(raw_energies_0)
    total_uncertainties_1 = compute_total_uncertainties(raw_energies_1)

    total_uncertainties = (1 - lambda_value) * total_uncertainties_0 + lambda_value * total_uncertainties_1

    linear_penalty = np.maximum(0, total_uncertainties - total_thresh)
    return linear_penalty

def compute_last_valid_ind(linear_penalty):
    last_valid_ind = np.argmax(np.cumsum(linear_penalty) > 0) - 1
    return last_valid_ind

from collections import namedtuple
RawEnergies = namedtuple('RawEnergies',
                         ['raw_energies_without_dummy_0',
                          'raw_energies_without_dummy_1'
                         ]
                        )

try:
    from pickle import load
    with open('raw_energy_dict.pkl', 'rb') as f:
        raw_energy_dict = load(f)
except:
    raw_energy_dict = {}
    for dcd in dcds:
        print(dcd)
        lam = parse_lambda_from_filename(dcd)
        traj = md.load_dcd(dcd, top=tautomer.ligand_in_water_topology)
        raw_energies_without_dummy_0, raw_energies_without_dummy_1 = compute_endstate_atomic_energy_contributions(traj[::10], model, energy_function)
        raw_energies = RawEnergies(raw_energies_without_dummy_0, raw_energies_without_dummy_1)
        raw_energy_dict[lam] = raw_energies
    from pickle import dump
    with open('raw_energy_dict.pkl', 'wb') as f:
        dump(raw_energy_dict, f)

trajs = {}
for dcd in dcds:
    lam = parse_lambda_from_filename(dcd)
    print(lam)
    traj = md.load_dcd(dcd, top=tautomer.ligand_in_water_topology)[::10]
    trajs[lam] = traj



last_valid_inds = {}
for lam in raw_energy_dict:
    raw_energies_0 = raw_energy_dict[lam].raw_energies_without_dummy_0
    raw_energies_1 = raw_energy_dict[lam].raw_energies_without_dummy_1

    linear_penalty = compute_linear_penalty(raw_energies_0, raw_energies_1, lam)
    last_valid_ind = compute_last_valid_ind(linear_penalty)
    last_valid_inds[lam] = last_valid_ind

lambdas_with_usable_samples = []
for lam in sorted(list(last_valid_inds.keys())):
    if last_valid_inds[lam] > 5:
        lambdas_with_usable_samples.append(lam)
print(lambdas_with_usable_samples)

snapshots = []
N_k = []

max_n_snapshots_per_state = 10

for lam in lambdas_with_usable_samples:
    traj = trajs[lam][5:last_valid_inds[lam]]
    further_thinning = 1
    if len(traj) > max_n_snapshots_per_state:
        further_thinning = int(len(traj) / max_n_snapshots_per_state)
    new_snapshots = list(traj.xyz[::further_thinning] * unit.nanometer)
    snapshots.extend(new_snapshots)
    N_k.append(len(new_snapshots))



N = len(snapshots)
print(N_k, N)

def compute_annealed_reduced_u(raw_energies, lambda_value=0.0):
    raw_energies_0 = raw_energies.raw_energies_without_dummy_0
    raw_energies_1 = raw_energies.raw_energies_without_dummy_1

    total_energies_0 = raw_energies_0.mean(-1).sum(1)
    total_energies_1 = raw_energies_1.mean(-1).sum(1)
    return (hartree_to_kJ_mol) * (
                (1 - lambda_value) * total_energies_0 + lambda_value * total_energies_1) * unit.kilojoule_per_mole / kT
