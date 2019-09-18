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
from neutromeratio.constants import device
import torchani

def test_equ():
    assert(1.0 == 1.0)

def test_neutromeratio_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "neutromeratio" in sys.modules

#def test_constants_init():

#    neutromeratio.initialize_variables(300, 'cpu')
#    assert(neutromeratio.constants.platform == 'cpu')
#    assert(type(neutromeratio.constants.temperature) == unit.Quantity)





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
    tautomer_transformation = neutromeratio.get_tautomer_transformation(from_mol, to_mol)
    ani_input = neutromeratio.from_mol_to_ani_input(from_mol, nr_of_conf=2)
    print(tautomer_transformation)
    atoms = ani_input['ligand_atoms']
    
    assert(atoms == 'CCCCCOOHHHHHHHH')
    assert(tautomer_transformation['hydrogen_idx'] == 11)
    assert(tautomer_transformation['acceptor_idx'] == 5)
    assert(tautomer_transformation['donor_idx'] == 2)


def test_neutromeratio_energy_calculations_with_torchANI_model():

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
    
    # generate tautomer transformation
    tautomer_transformation = neutromeratio.get_tautomer_transformation(from_mol, to_mol)

    # generate ani input
    ani_input = neutromeratio.from_mol_to_ani_input(from_mol, nr_of_conf=2)
    atoms = ani_input['ligand_atoms']

    # overwrite the coordinates that rdkit generated with the first frame in the traj
    x0 = traj[0]

    model = torchani.models.ANI1ccx()
    
    torch.set_num_threads(1)

    energy_function = neutromeratio.ANI1_force_and_energy(
                                                model = model,
                                                atoms = atoms)

    energy_function.flat_bottom_restraint  = False
    energy_function.harmonic_restraint  = False
    energy_function.use_pure_ani1ccx = True
    x = energy_function.calculate_energy(x0, lambda_value=0.0)
    x = x.value_in_unit(unit.kilocalorie_per_mole)
    np.testing.assert_almost_equal(x, -216736.6903680688, decimal=5)


def test_neutromeratio_energy_calculations_with_LinearAlchemicalANI_model():

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
    tautomer_transformation = neutromeratio.get_tautomer_transformation(from_mol, to_mol)
    
    # generate tautomer transformation
    hydrogen_idx = tautomer_transformation['hydrogen_idx']
    ani_input = neutromeratio.from_mol_to_ani_input(from_mol, nr_of_conf=2)
    atoms = ani_input['ligand_atoms']

    # overwrite the coordinates that rdkit generated with the first frame in the traj
    x0 = traj[0]

    model = neutromeratio.ani.LinearAlchemicalANI(alchemical_atoms=[hydrogen_idx], ani_input=ani_input)
    
    torch.set_num_threads(1)

    energy_function = neutromeratio.ANI1_force_and_energy(
                                                model = model,
                                                atoms = atoms)

    energy_function.flat_bottom_restraint  = False
    energy_function.harmonic_restraint  = False
    energy_function.use_pure_ani1ccx = False
    x = energy_function.calculate_energy(x0, lambda_value=1.0)
    x = x.value_in_unit(unit.kilocalorie_per_mole)
    np.testing.assert_almost_equal(x, -216736.6857518717, decimal=5)
    x = energy_function.calculate_energy(x0, lambda_value=0.0)
    x = x.value_in_unit(unit.kilocalorie_per_mole)
    np.testing.assert_almost_equal(x, -216698.911373172, decimal=5)


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


    tautomer_transformation = neutromeratio.get_tautomer_transformation(from_mol, to_mol)
    ani_input = neutromeratio.from_mol_to_ani_input(from_mol, nr_of_conf=2)
    neutromeratio.generate_hybrid_structure(ani_input, tautomer_transformation)
    assert(ani_input['hybrid_atoms'] == 'OCCCNCCCCCCHHHHHHHH')
    
    ani_traj = md.Trajectory(ani_input['hybrid_coords'].value_in_unit(unit.nanometer), ani_input['hybrid_topology'])
    e = ani_input['min_e']
    
    assert(type(e) == unit.Quantity)
    assert(type(tautomer_transformation['donor_hydrogen_idx']) == int)
    e = e.value_in_unit(unit.kilocalorie_per_mole)
    assert(e < -216698.91137612148 + 5)


def test_restraint():

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
    tautomer_transformation = neutromeratio.get_tautomer_transformation(from_mol, to_mol)
    ani_input = neutromeratio.from_mol_to_ani_input(from_mol, nr_of_conf=2)
    atoms = ani_input['ligand_atoms']
    print(atoms)
    print(ani_input['ligand_coords'])
    a = neutromeratio.Restraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms, active_at_lambda=-1)
    coordinates = torch.tensor([ani_input['ligand_coords'][0].value_in_unit(unit.nanometer)],
                        requires_grad=True, device=device, dtype=torch.float32)

    print('Restraing: {}.'.format(a.harmonic_position_restraint(coordinates)))
    print('Restraing: {}.'.format(a.flat_bottom_position_restraint(coordinates)))


def test_restraint_with_alchemicalANI():

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
    
    # generate tautomer transformation
    tautomer_transformation = neutromeratio.get_tautomer_transformation(from_mol, to_mol)

    # generate ani input 
    ani_input = neutromeratio.from_mol_to_ani_input(from_mol, nr_of_conf=2)

    atoms = ani_input['ligand_atoms']
    hydrogen_idx = tautomer_transformation['hydrogen_idx']

    # overwrite the coordinates that rdkit generated with the first frame in the traj
    x0 = traj[0]

    model = neutromeratio.ani.LinearAlchemicalANI(alchemical_atoms=[hydrogen_idx], ani_input=ani_input)
    
    energy_function = neutromeratio.ANI1_force_and_energy(
                                                model = model,
                                                atoms = atoms)

    energy_function.flat_bottom_restraint  = False
    energy_function.harmonic_restraint  = False
    energy_function.use_pure_ani1ccx = False
    x = energy_function.calculate_energy(x0, lambda_value=1.0)
    x = x.value_in_unit(unit.kilocalorie_per_mole)
    np.testing.assert_almost_equal(x, -216736.6857518717, decimal=5)
    x = energy_function.calculate_energy(x0, lambda_value=0.0)
    x = x.value_in_unit(unit.kilocalorie_per_mole)
    np.testing.assert_almost_equal(x, -216698.911373172, decimal=5)

    # add the restraints - active at different lambda steps
    restrain1 = neutromeratio.Restraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms, active_at_lambda=0)
    restrain2 = neutromeratio.Restraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms, active_at_lambda=1)
    restrain3 = neutromeratio.Restraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms, active_at_lambda=-1)

    energy_function.add_restraint(restrain1)
    energy_function.add_restraint(restrain2)
    energy_function.add_restraint(restrain3)

    # test for that restraints do not add to energy
    energy_function.flat_bottom_restraint  = False
    energy_function.harmonic_restraint  = False
    energy_function.use_pure_ani1ccx = False

    x = energy_function.calculate_energy(x0, lambda_value=0.0)
    x = x.value_in_unit(unit.kilocalorie_per_mole)
    np.testing.assert_almost_equal(x, -216698.911373172, decimal=5)

    # test flat_bottom_restraint for lambda = 0.0
    energy_function.flat_bottom_restraint  = True
    energy_function.harmonic_restraint  = False
    energy_function.use_pure_ani1ccx = False

    x = energy_function.calculate_energy(x0, lambda_value=0.0)
    x = x.value_in_unit(unit.kilocalorie_per_mole)
    np.testing.assert_almost_equal(x, -216527.22548065928, decimal=5)

    # test harmonic_restraint for lambda = 0.0 
    energy_function.flat_bottom_restraint  = False
    energy_function.harmonic_restraint  = True
    energy_function.use_pure_ani1ccx = False

    x = energy_function.calculate_energy(x0, lambda_value=0.0)
    x = x.value_in_unit(unit.kilocalorie_per_mole)
    np.testing.assert_almost_equal(x, -216484.37304754654, decimal=5)

    # test harmonic_restraint for lambda = 1.0 
    energy_function.flat_bottom_restraint  = False
    energy_function.harmonic_restraint  = True
    energy_function.use_pure_ani1ccx = False

    x = energy_function.calculate_energy(x0, lambda_value=1.0)
    x = x.value_in_unit(unit.kilocalorie_per_mole)
    np.testing.assert_almost_equal(x, -216522.14742624626, decimal=5)

    # test harmonic_restraint and flat_bottom_restraint for lambda = 1.0 
    energy_function.flat_bottom_restraint  = True
    energy_function.harmonic_restraint  = True
    energy_function.use_pure_ani1ccx = False

    x = energy_function.calculate_energy(x0, lambda_value=1.0)
    x = x.value_in_unit(unit.kilocalorie_per_mole)
    np.testing.assert_almost_equal(x, -216350.46153373353, decimal=5)


def test_restraint_with_alchemicalANISingleTopology():
  
    # read in exp_results.pickle
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    # generate smiles
    name = 'molDWRow_298'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    # define the direction of the tautomer reaction
    from_mol_tautomer_idx = 1
    to_mol_tautomer_idx = 2

    # generate both rdkit mol
    mols = { 't1' : neutromeratio.generate_rdkit_mol(t1_smiles), 't2' : neutromeratio.generate_rdkit_mol(t2_smiles) }
    from_mol = mols[f"t{from_mol_tautomer_idx}"]
    to_mol = mols[f"t{to_mol_tautomer_idx}"]
    
    # generate tautomer transformation
    tautomer_transformation = neutromeratio.get_tautomer_transformation(from_mol, to_mol)

    # generate ani input 
    ani_input = neutromeratio.from_mol_to_ani_input(from_mol, nr_of_conf=2)

    # generate hybrid input
    neutromeratio.generate_hybrid_structure(ani_input, tautomer_transformation)
    atoms = ani_input['hybrid_atoms']

    # rdkit coordinates
    x0 = ani_input['hybrid_coords']

    # the first of the alchemical_atoms will be dummy at lambda 0, the second at lambda 1
    # protocoll goes from 0 to 1
    dummy_atoms=[tautomer_transformation['acceptor_hydrogen_idx'], tautomer_transformation['donor_hydrogen_idx']]

    model = neutromeratio.ani.LinearAlchemicalSingleTopologyANI(alchemical_atoms=dummy_atoms, ani_input=ani_input)
    
    energy_function = neutromeratio.ANI1_force_and_energy(
                                                model = model,
                                                atoms = atoms)

    energy_function.flat_bottom_restraint  = False
    energy_function.harmonic_restraint  = False
    energy_function.use_pure_ani1ccx = False
    x = energy_function.calculate_energy(x0, lambda_value=0.0)
    x = x.value_in_unit(unit.kilocalorie_per_mole)


def test_euqilibrium():
    # name of the system
    name = 'molDWRow_298'
    # number of steps
    n_steps = 50

    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))

    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    # generate both rdkit mol
    mols = { 't1' : neutromeratio.generate_rdkit_mol(t1_smiles), 't2' : neutromeratio.generate_rdkit_mol(t2_smiles) }
    from_mol = mols['t1']
    to_mol = mols['t2']
    ani_input = neutromeratio.from_mol_to_ani_input(from_mol, nr_of_conf=1)

    tautomer_transformation = neutromeratio.get_tautomer_transformation(from_mol, to_mol)
    neutromeratio.generate_hybrid_structure(ani_input, tautomer_transformation)

    # define the alchemical atoms
    alchemical_atoms=[tautomer_transformation['acceptor_hydrogen_idx'], tautomer_transformation['donor_hydrogen_idx']]

    np.random.seed(0)

    # extract hydrogen donor idx and hydrogen idx for from_mol
    model = neutromeratio.ani.LinearAlchemicalSingleTopologyANI(alchemical_atoms=alchemical_atoms, ani_input=ani_input)
    model = model.to(device)
    torch.set_num_threads(2)

    # perform initial sampling
    energy_function = neutromeratio.ANI1_force_and_energy(
                                            model = model,
                                            atoms = ani_input['hybrid_atoms'],
                                            )

    for e in ani_input['hybrid_restraints']:
        energy_function.add_restraint(e)

    x0 = np.array(ani_input['hybrid_coords']) * unit.angstrom



    energy_function.minimize(ani_input)

    lambda_value = 1.0
    energy_and_force = lambda x : energy_function.calculate_force(x, lambda_value)

    langevin = neutromeratio.LangevinDynamics(atoms = ani_input['hybrid_atoms'],
                                temperature = 300*unit.kelvin,
                                energy_and_force = energy_and_force)

    equilibrium_samples, energies = langevin.run_dynamics(x0, n_steps=n_steps, stepsize=1.0 * unit.femtosecond, progress_bar=True)

    lambda_value = 0.0
    energy_and_force = lambda x : energy_function.calculate_force(x, lambda_value)

    langevin = neutromeratio.LangevinDynamics(atoms = ani_input['hybrid_atoms'],
                                temperature = 300*unit.kelvin,
                                energy_and_force = energy_and_force)

    equilibrium_samples, energies = langevin.run_dynamics(x0, n_steps=n_steps, stepsize=1.0 * unit.femtosecond, progress_bar=True)

