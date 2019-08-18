"""
Unit and regression test for the neutromeratio package.
"""

# Import package, test suite, and other packages as needed
import neutromeratio
import pytest
import sys
import pickle
import torch
from simtk import unit
import numpy as np

def test_neutromeratio_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "neutromeratio" in sys.modules

def test_tautomer_transformation():

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
    atom_list = ani_input['atom_list']
    
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
    atom_list = ani_input['atom_list']
    hydrogen_idx = tautomer_transformation['hydrogen_idx']

    # overwrite the coordinates that rdkit generated with the first frame in the traj
    x0 = traj[0]

    platform = 'cpu'
    device = torch.device(platform)
    model = neutromeratio.ani.LinearAlchemicalANI(alchemical_atom=hydrogen_idx)
    model = model.to(device)
    torch.set_num_threads(1)

    energy_function = neutromeratio.ANI1_force_and_energy(device = device,
                                                model = model,
                                                atom_list = atom_list,
                                                platform = platform,
                                                tautomer_transformation = tautomer_transformation)
    
    x = energy_function.calculate_energy(x0)
    x = x.value_in_unit(unit.kilocalorie_per_mole)
    assert(np.round(x, 2) == np.round(-216736.6857480091, 2))


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
    atom_list = ani_input['atom_list']
    hydrogen_idx = tautomer_transformation['hydrogen_idx']

    # overwrite the coordinates that rdkit generated with the first frame in the traj
    x0 = traj[0]

    platform = 'cpu'
    device = torch.device(platform)
    model = neutromeratio.ani.LinearAlchemicalANI(alchemical_atom=hydrogen_idx)
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
    assert(np.round(x, 2) == np.round(-216698.91137612148, 2) )