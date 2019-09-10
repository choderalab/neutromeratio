"""
Unit and regression test for the neutromeratio package.
"""

# Import package, test suite, and other packages as needed
import neutromeratio
import pytest
import sys, os
import pickle
import torch
from simtk import unit
import numpy as np
import mdtraj as md

def test_equ():
    assert(1.0 == 1.0)

def test_neutromeratio_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "neutromeratio" in sys.modules

def test_tautomer_transformation():

    print(os.getcwd())
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    name = 'molDWRow_298'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    from_mol_tautomer_idx = 1
    to_mol_tautomer_idx = 2

    # generate both rdkit mol
    mols = { 't1' : neutromeratio.generate_rdkit_mol(t1_smiles), 't2' : neutromeratio.generate_rdkit_mol(t2_smiles) }
    from_mol = mols[f"t{from_mol_tautomer_idx}"]
    to_mol = mols[f"t{to_mol_tautomer_idx}"]
    ani_input = neutromeratio.from_mol_to_ani_input(from_mol)
    tautomer_transformation = neutromeratio.get_tautomer_transformation(from_mol, to_mol)
    atom_list = ani_input['ligand_atoms']
    
    assert(atom_list == 'CCCCCOOHHHHHHHH')
    assert(tautomer_transformation['hydrogen_idx'] == 11)
    assert(tautomer_transformation['acceptor_idx'] == 5)
    assert(tautomer_transformation['donor_idx'] == 2)


def test_neutromeratio_energy_calculations():

    # read in pregenerated traj
    traj = neutromeratio.equilibrium.read_precalculated_md('neutromeratio/data/structure.pdb', 'neutromeratio/data/structure.dcd')
    
    # read in exp_results.pickle
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    # generate smiles
    name = 'molDWRow_298'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    from_mol_tautomer_idx = 1
    to_mol_tautomer_idx = 2

    # generate both rdkit mol
    mols = { 't1' : neutromeratio.generate_rdkit_mol(t1_smiles), 't2' : neutromeratio.generate_rdkit_mol(t2_smiles) }
    from_mol = mols[f"t{from_mol_tautomer_idx}"]
    to_mol = mols[f"t{to_mol_tautomer_idx}"]
    ani_input = neutromeratio.from_mol_to_ani_input(from_mol)
    
    # generate tautomer transformation
    tautomer_transformation = neutromeratio.get_tautomer_transformation(from_mol, to_mol)
    atom_list = ani_input['ligand_atoms']
    hydrogen_idx = tautomer_transformation['hydrogen_idx']

    # overwrite the coordinates that rdkit generated with the first frame in the traj
    x0 = traj[0]

    platform = 'cpu'
    device = torch.device(platform)
    model = neutromeratio.ani.LinearAlchemicalANI(alchemical_atoms=[hydrogen_idx], ani_input=ani_input, device=device)
    model = model.to(device)
    torch.set_num_threads(1)

    energy_function = neutromeratio.ANI1_force_and_energy(device = device,
                                                model = model,
                                                atom_list = atom_list,
                                                platform = platform,
                                                tautomer_transformation = tautomer_transformation)
    
    x = energy_function.calculate_energy(x0)
    x = x.value_in_unit(unit.kilocalorie_per_mole)
    assert(x == -216736.68575041983)


def test_neutromeratio_energy_calculations_with_dummy_atom():

    # read in pregenerated traj
    traj = neutromeratio.equilibrium.read_precalculated_md('neutromeratio/data/structure.pdb', 'neutromeratio/data/structure.dcd')
    
    # read in exp_results.pickle
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    # generate smiles
    name = 'molDWRow_298'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    from_mol_tautomer_idx = 1
    to_mol_tautomer_idx = 2

    # generate both rdkit mol
    mols = { 't1' : neutromeratio.generate_rdkit_mol(t1_smiles), 't2' : neutromeratio.generate_rdkit_mol(t2_smiles) }
    from_mol = mols[f"t{from_mol_tautomer_idx}"]
    to_mol = mols[f"t{to_mol_tautomer_idx}"]
    ani_input = neutromeratio.from_mol_to_ani_input(from_mol)
    
    # generate tautomer transformation
    tautomer_transformation = neutromeratio.get_tautomer_transformation(from_mol, to_mol)
    atom_list = ani_input['ligand_atoms']
    hydrogen_idx = tautomer_transformation['hydrogen_idx']

    # overwrite the coordinates that rdkit generated with the first frame in the traj
    x0 = traj[0]

    platform = 'cpu'
    device = torch.device(platform)
    model = neutromeratio.ani.LinearAlchemicalANI(alchemical_atoms=[hydrogen_idx], ani_input=ani_input, device=device)
    model = model.to(device)
    torch.set_num_threads(1)

    energy_function = neutromeratio.ANI1_force_and_energy(device = device,
                                                model = model,
                                                atom_list = atom_list,
                                                platform = platform,
                                                tautomer_transformation = tautomer_transformation)
    
    energy_function.lambda_value = 0.0

    x = energy_function.calculate_energy(x0)
    x = x.value_in_unit(unit.kilocalorie_per_mole)
    assert(x == -216698.91137287067)


def test_hybrid_topology():

    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    # specify the system you want to simulate
    name = 'molDWRow_590'

    from_mol_tautomer_idx = 1
    to_mol_tautomer_idx = 2

    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    # generate both rdkit mol
    mols = { 't1' : neutromeratio.generate_rdkit_mol(t1_smiles), 't2' : neutromeratio.generate_rdkit_mol(t2_smiles) }
    from_mol = mols[f"t{from_mol_tautomer_idx}"]
    to_mol = mols[f"t{to_mol_tautomer_idx}"]
    ani_input = neutromeratio.from_mol_to_ani_input(from_mol)

    tautomer_transformation = neutromeratio.get_tautomer_transformation(from_mol, to_mol)
    neutromeratio.generate_hybrid_structure(ani_input, tautomer_transformation, neutromeratio.ani.ANI1_force_and_energy)
    assert(ani_input['hybrid_atoms'] == 'OCCCNCCCCCCHHHHHHHH')
    ani_traj = md.Trajectory(ani_input['hybrid_coords'].value_in_unit(unit.nanometer), ani_input['hybrid_topolog'])
    e = ani_input['min_e']
    
    assert(type(e) == unit.Quantity)
    assert(type(tautomer_transformation['donor_hydrogen_idx']) == int)
    e = e.value_in_unit(unit.kilocalorie_per_mole)
    assert(e < -216698.91137612148 + 5)
