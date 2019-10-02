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
from ..utils import is_quantity_close

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
    assert(t.hybrid_hydrogen_idx_at_donor_heavy_atom == 11)
    assert(t.hybrid_hydrogen_idx_at_acceptor_heavy_atom == 15)

    t.perform_tautomer_transformation_reverse()

    # test if t1 centric mapping of hydrogen, heavy atom acceptor and donor works
    assert(t.intial_state_ligand_atoms == 'CCCCCOOHHHHHHHH')
    assert(t.hydrogen_idx == 14)
    assert(t.heavy_atom_hydrogen_acceptor_idx  == 2)
    assert(t.heavy_atom_hydrogen_donor_idx == 5)

    # test if dual topology hybrid works
    assert(t.hybrid_atoms == 'CCCCCOOHHHHHHHHH')
    assert(t.hybrid_hydrogen_idx_at_donor_heavy_atom == 14)
    assert(t.hybrid_hydrogen_idx_at_acceptor_heavy_atom == 15)


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
    assert(t.hybrid_hydrogen_idx_at_donor_heavy_atom == 18)
    assert(t.hybrid_hydrogen_idx_at_acceptor_heavy_atom == 19)

    t.perform_tautomer_transformation_reverse()
    
    # test if t2 centric mapping of hydrogen, heavy atom acceptor and donor works
    assert(t.intial_state_ligand_atoms == 'CCCCCCCOOHHHHHHHHHH')
    assert(t.hydrogen_idx == 12)
    assert(t.heavy_atom_hydrogen_acceptor_idx  == 8)
    assert(t.heavy_atom_hydrogen_donor_idx == 2)

    # test if dual topology hybrid works CCOCCCCCOHHHHHHHHHHH
    assert(t.hybrid_atoms == 'CCOCCCCCOHHHHHHHHHHH')
    assert(t.hybrid_hydrogen_idx_at_donor_heavy_atom == 12)
    assert(t.hybrid_hydrogen_idx_at_acceptor_heavy_atom == 19)


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

    energy_function.flat_bottom_restraint  = False
    energy_function.harmonic_restraint  = False
    energy_function.use_pure_ani1ccx = True
    x = energy_function.calculate_energy(x0,)
    assert(is_quantity_close(x, -216736.6903680688 * unit.kilocalorie_per_mole, rtol=1e-9))


    # testint reverse - it should get the same energy
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

    energy_function.flat_bottom_restraint  = False
    energy_function.harmonic_restraint  = False
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

    energy_function.flat_bottom_restraint  = False
    energy_function.harmonic_restraint  = False
    energy_function.use_pure_ani1ccx = False
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
    dummy_atoms = [t.hybrid_hydrogen_idx_at_acceptor_heavy_atom, t.hybrid_hydrogen_idx_at_donor_heavy_atom]
    atoms = t.hybrid_atoms
    # overwrite the coordinates that rdkit generated with the first frame in the traj
    model = neutromeratio.ani.LinearAlchemicalDualTopologyANI(alchemical_atoms=dummy_atoms)
    torch.set_num_threads(1)

    energy_function = neutromeratio.ANI1_force_and_energy(
                                                model = model,
                                                atoms = atoms,
                                                mol = t.hybrid_ase_mol)
    
    traj = neutromeratio.equilibrium.read_precalculated_md('neutromeratio/data/mol298_t1.pdb', 'neutromeratio/data/mol298_t1.dcd')

    x0 = traj[0]
    energy_function.flat_bottom_restraint  = False
    energy_function.harmonic_restraint  = False
    energy_function.use_pure_ani1ccx = False
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

    t = Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles))
    t.perform_tautomer_transformation_forward()

    atoms = t.intial_state_ligand_atoms
    a = neutromeratio.Restraint(sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms, active_at_lambda=-1)
    
    coordinates = torch.tensor([t.intial_state_ligand_coords[0].value_in_unit(unit.nanometer)],
                        requires_grad=True, device=device, dtype=torch.float32)

    print('Restraing: {}.'.format(a.harmonic_position_restraint(coordinates)))
    print('Restraing: {}.'.format(a.flat_bottom_position_restraint(coordinates)))


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

    t = Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles))
    t.perform_tautomer_transformation_forward()

    atoms = t.intial_state_ligand_atoms
    hydrogen_idx = t.hydrogen_idx

    # overwrite the coordinates that rdkit generated with the first frame in the traj
    x0 = traj[0]

    model = neutromeratio.ani.LinearAlchemicalANI(alchemical_atoms=[hydrogen_idx])
    
    energy_function = neutromeratio.ANI1_force_and_energy(
                                                model = model,
                                                atoms = atoms,
                                                mol = t.intial_state_ase_mol)

    energy_function.flat_bottom_restraint  = False
    energy_function.harmonic_restraint  = False
    energy_function.use_pure_ani1ccx = False
    x = energy_function.calculate_energy(x0, lambda_value=1.0)
    assert(is_quantity_close(x, -216736.6857518717* unit.kilocalorie_per_mole, rtol=1e-9))
    x = energy_function.calculate_energy(x0, lambda_value=0.0)
    assert(is_quantity_close(x, -216698.911373172* unit.kilocalorie_per_mole, rtol=1e-9))

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
    assert(is_quantity_close(x, -216698.911373172* unit.kilocalorie_per_mole, rtol=1e-9))

    # test flat_bottom_restraint for lambda = 0.0
    energy_function.flat_bottom_restraint  = True
    energy_function.harmonic_restraint  = False
    energy_function.use_pure_ani1ccx = False

    x = energy_function.calculate_energy(x0, lambda_value=0.0)
    assert(is_quantity_close(x, -216527.22548065928* unit.kilocalorie_per_mole, rtol=1e-9))

    # test harmonic_restraint for lambda = 0.0 
    energy_function.flat_bottom_restraint  = False
    energy_function.harmonic_restraint  = True
    energy_function.use_pure_ani1ccx = False

    x = energy_function.calculate_energy(x0, lambda_value=0.0)
    assert(is_quantity_close(x, -216484.37304754654* unit.kilocalorie_per_mole, rtol=1e-9))

    # test harmonic_restraint for lambda = 1.0 
    energy_function.flat_bottom_restraint  = False
    energy_function.harmonic_restraint  = True
    energy_function.use_pure_ani1ccx = False

    x = energy_function.calculate_energy(x0, lambda_value=1.0)
    assert(is_quantity_close(x, -216522.14742624626* unit.kilocalorie_per_mole, rtol=1e-9))

    # test harmonic_restraint and flat_bottom_restraint for lambda = 1.0 
    energy_function.flat_bottom_restraint  = True
    energy_function.harmonic_restraint  = True
    energy_function.use_pure_ani1ccx = False

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

    t = neutromeratio.Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles))
    t.perform_tautomer_transformation_forward()
    
    traj = neutromeratio.equilibrium.read_precalculated_md('neutromeratio/data/mol298_t1.pdb', 'neutromeratio/data/mol298_t1.dcd')
    x0 = traj[0]

    # the first of the alchemical_atoms will be dummy at lambda 0, the second at lambda 1
    # protocoll goes from 0 to 1
    dummy_atoms = [t.hybrid_hydrogen_idx_at_acceptor_heavy_atom, t.hybrid_hydrogen_idx_at_donor_heavy_atom]
    atoms = t.hybrid_atoms

    model = neutromeratio.ani.LinearAlchemicalDualTopologyANI(alchemical_atoms=dummy_atoms)
    
    energy_function = neutromeratio.ANI1_force_and_energy(
                                                model = model,
                                                atoms = atoms,
                                                mol = t.hybrid_ase_mol)

    energy_function.flat_bottom_restraint  = False
    energy_function.harmonic_restraint  = False
    energy_function.use_pure_ani1ccx = False
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
            x = energy_function.minimize(coords)

            x0 = np.array(x) * unit.angstrom
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
            x = energy_function.minimize(coords)
            energy_function.get_thermo_correction(x)



def test_euqilibrium():
    # name of the system
    name = 'molDWRow_298'
    # number of steps
    n_steps = 50

    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))

    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t = neutromeratio.Tautomer(name=name, intial_state_mol=neutromeratio.generate_rdkit_mol(t1_smiles), final_state_mol=neutromeratio.generate_rdkit_mol(t2_smiles))
    t.perform_tautomer_transformation_forward()

    # define the alchemical atoms
    alchemical_atoms=[t.hybrid_hydrogen_idx_at_acceptor_heavy_atom, t.hybrid_hydrogen_idx_at_donor_heavy_atom]

    # extract hydrogen donor idx and hydrogen idx for from_mol
    model = neutromeratio.ani.LinearAlchemicalDualTopologyANI(alchemical_atoms=alchemical_atoms)
    model = model.to(device)
    torch.set_num_threads(2)

    # perform initial sampling
    energy_function = neutromeratio.ANI1_force_and_energy(
                                            model = model,
                                            atoms = t.hybrid_atoms,
                                            mol = t.hybrid_ase_mol
                                            )

    for e in t.hybrid_restraints:
        energy_function.add_restraint(e)

    x0 = np.array(t.hybrid_coords) * unit.angstrom

    x = energy_function.minimize(x0)

    lambda_value = 1.0
    energy_and_force = lambda x : energy_function.calculate_force(x, lambda_value)

    langevin = neutromeratio.LangevinDynamics(atoms = t.hybrid_atoms,
                                temperature = 300*unit.kelvin,
                                energy_and_force = energy_and_force,
                                )

    equilibrium_samples, energies = langevin.run_dynamics(x0, n_steps=n_steps, stepsize=1.0 * unit.femtosecond, progress_bar=True)

    lambda_value = 0.0
    energy_and_force = lambda x : energy_function.calculate_force(x, lambda_value)

    langevin = neutromeratio.LangevinDynamics(atoms = t.hybrid_atoms,
                                temperature = 300*unit.kelvin,
                                energy_and_force = energy_and_force)

    equilibrium_samples, energies = langevin.run_dynamics(x0, n_steps=n_steps, stepsize=1.0 * unit.femtosecond, progress_bar=True)

