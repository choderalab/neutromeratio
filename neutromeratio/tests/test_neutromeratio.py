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
from openmmtools.utils import is_quantity_close
import pandas as pd

def test_equ():
    assert(1.0 == 1.0)

def test_neutromeratio_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "neutromeratio" in sys.modules

#def test_constants_init():

#    neutromeratio.initialize_variables(300, 'cpu')
#    assert(neutromeratio.constants.platform == 'cpu')
#    assert(type(neutromeratio.constants.temperature) == unit.Quantity)


def test_tautomer_class():

    from neutromeratio.tautomers import Tautomer
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
    t = Tautomer(name=name, intial_state_mol=from_mol, final_state_mol=to_mol)
    t.perform_tautomer_transformation_forward()
    


def test_tautomer_transformation():
    from neutromeratio.tautomers import Tautomer

    print(os.getcwd())
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    name = 'molDWRow_298'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']
   
    t = Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles))
    t.perform_tautomer_transformation_forward()

    
    # test if t1 centric mapping of hydrogen, heavy atom acceptor and donor works
    assert(t.intial_state_ligand_atoms == 'CCCCCOOHHHHHHHH')
    assert(t.hydrogen_idx == 11)
    assert(t.heavy_atom_hydrogen_acceptor_idx  == 5)
    assert(t.heavy_atom_hydrogen_donor_idx == 2)

    # test if dual topology hybrid works
    assert(t.hybrid_atoms == 'CCCCCOOHHHHHHHHH')
    assert(t.hydrogen_idx == 11)
    assert(t.hybrid_dummy_hydrogen == 15)

    t.perform_tautomer_transformation_reverse()

    # test if t1 centric mapping of hydrogen, heavy atom acceptor and donor works
    assert(t.intial_state_ligand_atoms == 'CCCCCOOHHHHHHHH')
    assert(t.hydrogen_idx == 14)
    assert(t.heavy_atom_hydrogen_acceptor_idx  == 2)
    assert(t.heavy_atom_hydrogen_donor_idx == 5)

    # test if dual topology hybrid works
    assert(t.hybrid_atoms == 'CCCCCOOHHHHHHHHH')
    assert(t.hydrogen_idx == 14)
    assert(t.hybrid_dummy_hydrogen == 15)


    name = 'molDWRow_37'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']
   
    t = Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles))
    t.perform_tautomer_transformation_forward()
    
    # test if t1 centric mapping of hydrogen, heavy atom acceptor and donor works
    assert(t.intial_state_ligand_atoms == 'CCCCCCCOOHHHHHHHHHH')
    assert(t.hydrogen_idx == 18)
    assert(t.heavy_atom_hydrogen_acceptor_idx  == 8)
    assert(t.heavy_atom_hydrogen_donor_idx == 7)

    # test if dual topology hybrid works
    assert(t.hybrid_atoms == 'CCCCCCCOOHHHHHHHHHHH')
    assert(t.hydrogen_idx == 18)
    assert(t.hybrid_dummy_hydrogen == 19)

    t.perform_tautomer_transformation_reverse()
    
    # test if t2 centric mapping of hydrogen, heavy atom acceptor and donor works
    assert(t.intial_state_ligand_atoms == 'CCCCCCCOOHHHHHHHHHH')
    assert(t.hydrogen_idx == 12)
    assert(t.heavy_atom_hydrogen_acceptor_idx  == 8)
    assert(t.heavy_atom_hydrogen_donor_idx == 2)

    # test if dual topology hybrid works CCOCCCCCOHHHHHHHHHHH
    assert(t.hybrid_atoms == 'CCOCCCCCOHHHHHHHHHHH')
    assert(t.hydrogen_idx == 12)
    assert(t.hybrid_dummy_hydrogen == 19)

    # test if droplet works for 
    t.add_droplet(t.final_state_ligand_topology, t.final_state_ligand_coords[0])

def test_neutromeratio_energy_calculations_with_torchANI_model():
    
    from neutromeratio.tautomers import Tautomer

    # read in pregenerated traj
    traj = neutromeratio.equilibrium.read_precalculated_md('neutromeratio/data/structure.pdb', 'neutromeratio/data/structure.dcd')
    
    # read in exp_results.pickle
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    # generate smiles
    name = 'molDWRow_298'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t = Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles))
    t.perform_tautomer_transformation_forward()
    atoms = t.intial_state_ligand_atoms

    # overwrite the coordinates that rdkit generated with the first frame in the traj
    x0 = traj[0]
    model = torchani.models.ANI1ccx()
    torch.set_num_threads(1)
    energy_function = neutromeratio.ANI1_force_and_energy(
                                                model = model,
                                                atoms = atoms,
                                                mol = t.intial_state_ase_mol)

    energy_function.use_pure_ani1ccx = True
    x = energy_function.calculate_energy(x0,)
    assert(is_quantity_close(x, -216736.6903680688 * unit.kilocalorie_per_mole, rtol=1e-9))


    # testing reverse - it should get the same energy
    # because we set the same coordinates and no bonded info 
    # is given

    t.perform_tautomer_transformation_reverse()
    # generate ani input
    atoms = t.final_state_ligand_atoms

    # overwrite the coordinates that rdkit generated with the first frame in the traj
    x0 = traj[0]
    model = torchani.models.ANI1ccx()
    torch.set_num_threads(1)
    energy_function = neutromeratio.ANI1_force_and_energy(
                                                model = model,
                                                atoms = atoms,
                                                mol = t.final_state_ase_mol)

    energy_function.use_pure_ani1ccx = True
    x = energy_function.calculate_energy(x0,)
    assert(is_quantity_close(x, -216736.6903680688 * unit.kilocalorie_per_mole, rtol=1e-9))


def test_neutromeratio_energy_calculations_with_LinearAlchemicalANI_model():
    from neutromeratio.tautomers import Tautomer

    # read in pregenerated traj
    traj = neutromeratio.equilibrium.read_precalculated_md('neutromeratio/data/structure.pdb', 'neutromeratio/data/structure.dcd')
    
    # read in exp_results.pickle
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    # generate smiles
    name = 'molDWRow_298'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t = Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles))
    t.perform_tautomer_transformation_forward()
  
    # generate tautomer transformation
    hydrogen_idx = t.hydrogen_idx
    atoms = t.intial_state_ligand_atoms

    # overwrite the coordinates that rdkit generated with the first frame in the traj
    x0 = traj[0]
    model = neutromeratio.ani.LinearAlchemicalANI(alchemical_atoms=[hydrogen_idx])
    torch.set_num_threads(1)

    energy_function = neutromeratio.ANI1_force_and_energy(
                                                model = model,
                                                atoms = atoms,
                                                mol = t.intial_state_ase_mol)

    x = energy_function.calculate_energy(x0, lambda_value=1.0)
    assert(is_quantity_close(x, -216736.6857518717* unit.kilocalorie_per_mole, rtol=1e-9))
    x = energy_function.calculate_energy(x0, lambda_value=0.0)
    assert(is_quantity_close(x, -216698.911373172* unit.kilocalorie_per_mole, rtol=1e-9))


def test_neutromeratio_energy_calculations_with_DualTopologyAlchemicalANI_model():
    from neutromeratio.tautomers import Tautomer
   
    # read in exp_results.pickle
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    # generate smiles
    name = 'molDWRow_298'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t = Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles))
    t.perform_tautomer_transformation_forward()
  
    # generate tautomer transformation
    dummy_atoms = [t.hybrid_dummy_hydrogen, t.hydrogen_idx]
    atoms = t.hybrid_atoms
    # overwrite the coordinates that rdkit generated with the first frame in the traj
    model = neutromeratio.ani.LinearAlchemicalDualTopologyANI(alchemical_atoms=dummy_atoms)
    torch.set_num_threads(1)

    energy_function = neutromeratio.ANI1_force_and_energy(
                                                model = model,
                                                atoms = atoms,
                                                mol = None)
    
    traj = neutromeratio.equilibrium.read_precalculated_md('neutromeratio/data/mol298_t1.pdb', 'neutromeratio/data/mol298_t1.dcd')

    x0 = traj[0]

    x = energy_function.calculate_energy(x0, lambda_value=1.0)
    assert(is_quantity_close(x, -216707.18481400612* unit.kilocalorie_per_mole, rtol=1e-9))
    x = energy_function.calculate_energy(x0, lambda_value=0.0)
    assert(is_quantity_close(x, -216763.81517969485* unit.kilocalorie_per_mole, rtol=1e-9))


def test_restraint():
    from neutromeratio.tautomers import Tautomer
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    name = 'molDWRow_298'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    tautomer = Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles))
    tautomer.perform_tautomer_transformation_forward()

    atoms = tautomer.intial_state_ligand_atoms
    harmonic = neutromeratio.restraints.HarmonicRestraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms)
    flat_bottom = neutromeratio.restraints.FlatBottomRestraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms)
    

    coordinates = torch.tensor([tautomer.intial_state_ligand_coords[0].value_in_unit(unit.nanometer)],
                        requires_grad=True, device=device, dtype=torch.float32)

    print('Restraing: {}.'.format(harmonic.restraint(coordinates)))
    print('Restraing: {}.'.format(flat_bottom.restraint(coordinates)))


def test_restraint_with_alchemicalANI():
    from neutromeratio.tautomers import Tautomer

    # read in pregenerated traj
    traj = neutromeratio.equilibrium.read_precalculated_md('neutromeratio/data/structure.pdb', 'neutromeratio/data/structure.dcd')
    
    # read in exp_results.pickle
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    # generate smiles
    name = 'molDWRow_298'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    tautomer = Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles))
    tautomer.perform_tautomer_transformation_forward()

    atoms = tautomer.intial_state_ligand_atoms
    hydrogen_idx = tautomer.hydrogen_idx

    # overwrite the coordinates that rdkit generated with the first frame in the traj
    x0 = traj[0]

    model = neutromeratio.ani.LinearAlchemicalANI(alchemical_atoms=[hydrogen_idx])
    
    energy_function = neutromeratio.ANI1_force_and_energy(
                                                model = model,
                                                atoms = atoms,
                                                mol = tautomer.intial_state_ase_mol)

    x = energy_function.calculate_energy(x0, lambda_value=1.0)
    assert(is_quantity_close(x, -216736.6857518717* unit.kilocalorie_per_mole, rtol=1e-9))
    x = energy_function.calculate_energy(x0, lambda_value=0.0)
    assert(is_quantity_close(x, -216698.911373172* unit.kilocalorie_per_mole, rtol=1e-9))


    # test flat_bottom_restraint for lambda = 0.0
    r = []
    restrain1 = neutromeratio.restraints.FlatBottomRestraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms, active_at_lambda=0)
    restrain2 = neutromeratio.restraints.FlatBottomRestraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms, active_at_lambda=1)
    restrain3 = neutromeratio.restraints.FlatBottomRestraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms, active_at_lambda=-1)
    for r in [restrain1, restrain2, restrain3]:
        energy_function.add_restraint(r)

    x = energy_function.calculate_energy(x0, lambda_value=0.0)
    assert(is_quantity_close(x, -216527.22548065928* unit.kilocalorie_per_mole, rtol=1e-9))

    # test harmonic_restraint for lambda = 0.0 
    energy_function.reset_restraints()
    r = []
    restrain1 = neutromeratio.restraints.HarmonicRestraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms, active_at_lambda=0)
    restrain2 = neutromeratio.restraints.HarmonicRestraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms, active_at_lambda=1)
    restrain3 = neutromeratio.restraints.HarmonicRestraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms, active_at_lambda=-1)
    for r in [restrain1, restrain2, restrain3]:
        energy_function.add_restraint(r)

    x = energy_function.calculate_energy(x0, lambda_value=0.0)
    assert(is_quantity_close(x, -216484.37304754654* unit.kilocalorie_per_mole, rtol=1e-9))

    # test harmonic_restraint for lambda = 1.0 
    x = energy_function.calculate_energy(x0, lambda_value=1.0)
    assert(is_quantity_close(x, -216522.14742624626* unit.kilocalorie_per_mole, rtol=1e-9))

    # test harmonic_restraint and flat_bottom_restraint for lambda = 1.0 
    r = []
    energy_function.reset_restraints()
    restrain1 = neutromeratio.restraints.FlatBottomRestraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms, active_at_lambda=0)
    restrain2 = neutromeratio.restraints.FlatBottomRestraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms, active_at_lambda=1)
    restrain3 = neutromeratio.restraints.FlatBottomRestraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms, active_at_lambda=-1)
    restrain4 = neutromeratio.restraints.HarmonicRestraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms, active_at_lambda=0)
    restrain5 = neutromeratio.restraints.HarmonicRestraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms, active_at_lambda=1)
    restrain6 = neutromeratio.restraints.HarmonicRestraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms, active_at_lambda=-1)
    for r in [restrain1, restrain2, restrain3, restrain4, restrain5, restrain6]:
        energy_function.add_restraint(r)

    x = energy_function.calculate_energy(x0, lambda_value=1.0)
    assert(is_quantity_close(x, -216350.46153373353* unit.kilocalorie_per_mole, rtol=1e-9))


def test_restraint_with_alchemicalANIDualTopology():
  
    # read in exp_results.pickle
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    # generate smiles
    name = 'molDWRow_298'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    tautomer = neutromeratio.Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles))
    tautomer.perform_tautomer_transformation_forward()
    
    traj = neutromeratio.equilibrium.read_precalculated_md('neutromeratio/data/mol298_t1.pdb', 'neutromeratio/data/mol298_t1.dcd')
    x0 = traj[0]

    # the first of the alchemical_atoms will be dummy at lambda 0, the second at lambda 1
    # protocoll goes from 0 to 1
    dummy_atoms = [tautomer.hybrid_dummy_hydrogen, tautomer.hydrogen_idx]
    atoms = tautomer.hybrid_atoms

    model = neutromeratio.ani.LinearAlchemicalDualTopologyANI(alchemical_atoms=dummy_atoms)
    
    energy_function = neutromeratio.ANI1_force_and_energy(
                                                model = model,
                                                atoms = atoms,
                                                mol = None)

    energy_function.list_of_restraints = tautomer.ligand_restraints

    x = energy_function.calculate_energy(x0, lambda_value=0.0)
    assert(is_quantity_close(x, -216763.81517969485* unit.kilocalorie_per_mole, rtol=1e-9))

def test_min_and_single_point_energy():
    
    # name of the system
    name = 'molDWRow_298'

    # extract smiles
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t = neutromeratio.Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles))

    # set model
    model = torchani.models.ANI1ccx()
    model = model.to(device)
    torch.set_num_threads(2)

    # calculate energy using both structures and pure ANI1ccx
    for ase_mol, ligand_atoms, ligand_coords in zip([t.intial_state_ase_mol, t.final_state_ase_mol], [t.intial_state_ligand_atoms, t.final_state_ligand_atoms], [t.intial_state_ligand_coords, t.final_state_ligand_coords]): 
        energy_function = neutromeratio.ANI1_force_and_energy(
                                                model = model,
                                                atoms = ligand_atoms,
                                                mol = ase_mol,
                                                use_pure_ani1ccx = True
                                            )

        for coords in ligand_coords:
            # minimize
            x0, hist_e = energy_function.minimize(coords)
            print(energy_function.calculate_energy(x0))


def test_thermochemistry():

    # name of the system
    name = 'molDWRow_298'

    # extract smiles
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    # generate both rdkit mol
    t = neutromeratio.Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles))
    t.perform_tautomer_transformation_forward()

    # set model
    model = torchani.models.ANI1ccx()
    model = model.to(device)
    torch.set_num_threads(1)

    # calculate energy using both structures and pure ANI1ccx
    for ase_mol, ligand_atoms, ligand_coords in zip([t.intial_state_ase_mol, t.final_state_ase_mol], [t.intial_state_ligand_atoms, t.final_state_ligand_atoms], [t.intial_state_ligand_coords, t.final_state_ligand_coords]): 
        energy_function = neutromeratio.ANI1_force_and_energy(
                                                model = model,
                                                atoms = ligand_atoms,
                                                mol = ase_mol,
                                                use_pure_ani1ccx = True
                                            )
        for coords in ligand_coords:
            # minimize
            x, hist_e = energy_function.minimize(coords)
            energy_function.get_thermo_correction(x)



def test_euqilibrium():
    # name of the system
    name = 'molDWRow_298'
    # number of steps
    n_steps = 50

    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))

    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    tautomer = neutromeratio.Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles))
    tautomer.perform_tautomer_transformation_forward()

    # define the alchemical atoms
    alchemical_atoms=[tautomer.hybrid_dummy_hydrogen, tautomer.hydrogen_idx]

    # extract hydrogen donor idx and hydrogen idx for from_mol
    model = neutromeratio.ani.LinearAlchemicalDualTopologyANI(alchemical_atoms=alchemical_atoms)
    model = model.to(device)
    torch.set_num_threads(2)

    # perform initial sampling
    energy_function = neutromeratio.ANI1_force_and_energy(
                                            model = model,
                                            atoms = tautomer.hybrid_atoms,
                                            mol = None
                                            )

    for e in tautomer.ligand_restraints + tautomer.hybrid_ligand_restraints:
        energy_function.add_restraint(e)

    x0 = np.array(tautomer.hybrid_coords) * unit.angstrom
    x0, hist_e = energy_function.minimize(x0)

    lambda_value = 1.0
    energy_and_force = lambda x : energy_function.calculate_force(x, lambda_value)

    langevin = neutromeratio.LangevinDynamics(atoms = tautomer.hybrid_atoms,
                                energy_and_force = energy_and_force,
                                )
    
    equilibrium_samples, energies, _ = langevin.run_dynamics(x0, n_steps=n_steps, stepsize=1.0 * unit.femtosecond, progress_bar=True)

    lambda_value = 0.0
    energy_and_force = lambda x : energy_function.calculate_force(x, lambda_value)

    langevin = neutromeratio.LangevinDynamics(atoms = tautomer.hybrid_atoms,
                                energy_and_force = energy_and_force)

    equilibrium_samples, energies, _ = langevin.run_dynamics(x0, n_steps=n_steps, stepsize=1.0 * unit.femtosecond, progress_bar=True)



def test_tautomer_conformation():
    # name of the system
    name = 'molDWRow_298'
    # number of steps

    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))

    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    tautomer = neutromeratio.Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles), nr_of_conformations=50)
    
    print(f"Nr of initial conformations: {tautomer.intial_state_ligand_coords}")
    print(f"Nr of final conformations: {tautomer.final_state_ligand_coords}")

    tautomer = neutromeratio.Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles), nr_of_conformations=5)   
    tautomer.perform_tautomer_transformation_forward()

    assert(len(tautomer.intial_state_ligand_coords) == 5)
    assert(len(tautomer.final_state_ligand_coords) == 5)


    # set model
    model = torchani.models.ANI1ccx()
    model = model.to(device)
    torch.set_num_threads(1)

    # calculate energy using both structures and pure ANI1ccx
    for ase_mol, ligand_atoms, ligand_coords in zip([tautomer.intial_state_ase_mol, tautomer.final_state_ase_mol], [tautomer.intial_state_ligand_atoms, tautomer.final_state_ligand_atoms], [tautomer.intial_state_ligand_coords, tautomer.final_state_ligand_coords]): 
        energy_function = neutromeratio.ANI1_force_and_energy(
                                                model = model,
                                                atoms = ligand_atoms,
                                                mol = ase_mol,
                                                use_pure_ani1ccx = True
                                            )
        for n_conf, coords in enumerate(ligand_coords):
            # minimize
            print(f"Conf: {n_conf}")
            x, e_min_history = energy_function.minimize(coords)
            e = energy_function.calculate_energy(x)
            e_correction = energy_function.get_thermo_correction(x)
            print(f"Energy: {e}")
            print(f"Energy correction: {e_correction}")
    

def test_mining_minima():
    # name of the system
    name = 'molDWRow_298'
    # number of steps
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
    torch.set_num_threads(1)

    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    tautomer = neutromeratio.Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles), nr_of_conformations=2)
    tautomer.perform_tautomer_transformation_forward()

    confs_traj, e, minimum_energies = tautomer.generate_mining_minima_structures()



def test_plotting():
    
    from neutromeratio.constants import kT
    from neutromeratio.plotting import plot_correlation_analysis
    results = pickle.load(open('neutromeratio/data/all_results.pickle', 'rb'))

    x_list = []
    y_list = []

    for a in list(results.ddG_DFT):
        a = a *kT
        a = a.value_in_unit(unit.kilocalorie_per_mole)
        x_list.append(a)
        
    for a in list(results.experimental_values):
        a = a *kT
        a = a.value_in_unit(unit.kilocalorie_per_mole)
        y_list.append(a)


    df = pd.DataFrame(list(zip(list(results.names), x_list, y_list, ['B3LYP/aug-cc-pVTZ']*len(results.names))), columns =['names', 'x', 'y', 'method']) 
    plot_correlation_analysis(df, 'DFT(B3LYP/aug-cc-pVTZ) in vacuum vs experimental data in solution', 'test1', 'test2', 'g', 'o')


def test_generating_droplet():

    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))

    name = 'molDWRow_298'

    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    # generate both rdkit mol
    tautomer = neutromeratio.Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles), nr_of_conformations=20)
    tautomer.perform_tautomer_transformation_forward()
    tautomer.add_droplet(tautomer.hybrid_topology, tautomer.hybrid_coords)

    # define the alchemical atoms
    alchemical_atoms=[tautomer.hybrid_dummy_hydrogen, tautomer.hydrogen_idx]

    # extract hydrogen donor idx and hydrogen idx for from_mol
    model = neutromeratio.ani.LinearAlchemicalDualTopologyANI(alchemical_atoms=alchemical_atoms)
    model = model.to(device)

    # perform initial sampling
    energy_function = neutromeratio.ANI1_force_and_energy(
                                            model = model,
                                            atoms = tautomer.ligand_in_water_atoms,
                                            mol = None,
                                            )

    for r in tautomer.ligand_restraints:
        energy_function.add_restraint(r)

    for r in tautomer.hybrid_ligand_restraints:
        energy_function.add_restraint(r)
        
