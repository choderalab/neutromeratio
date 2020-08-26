"""
Unit and regression test for the neutromeratio package.
"""

# Import package, test suite, and other packages as needed
import neutromeratio
import pytest
import sys
import os
import pickle
import torch
from simtk import unit
import numpy as np
import mdtraj as md
from neutromeratio.constants import device
import torchani
from openmmtools.utils import is_quantity_close
import pandas as pd
from rdkit import Chem

def test_equ():
    assert(1.0 == 1.0)

def test_neutromeratio_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "neutromeratio" in sys.modules

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
    mols = {'t1': neutromeratio.generate_rdkit_mol(t1_smiles), 't2': neutromeratio.generate_rdkit_mol(t2_smiles)}
    from_mol = mols[f"t{from_mol_tautomer_idx}"]
    to_mol = mols[f"t{to_mol_tautomer_idx}"]
    t = Tautomer(name=name, initial_state_mol=from_mol, final_state_mol=to_mol)
    t.perform_tautomer_transformation()




def test_tautomer_transformation():
    from neutromeratio.tautomers import Tautomer

    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    ###################################
    ###################################
    name = 'molDWRow_298'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles)
    t = tautomers[0]
    t.perform_tautomer_transformation()
    assert(len(tautomers) == 2)
    assert(neutromeratio.utils.get_stereotag_of_stereobonds(t.initial_state_mol) == 'STEREOZ')

    # test if t1 centric mapping of hydrogen, heavy atom acceptor and donor works
    assert(t.initial_state_ligand_atoms == 'CCCCCOOHHHHHHHH')
    assert(t.final_state_ligand_atoms == 'CCCCCOOHHHHHHHH')

    assert(t.hybrid_hydrogen_idx_at_lambda_0 == 14)
    assert(t.heavy_atom_hydrogen_acceptor_idx == 2)
    assert(t.heavy_atom_hydrogen_donor_idx == 5)

    # test if dual topology hybrid works
    assert(t.hybrid_atoms == 'CCCCCOOHHHHHHHHH')
    assert(t.hybrid_hydrogen_idx_at_lambda_0 == 14)
    assert(t.hybrid_hydrogen_idx_at_lambda_1 == 15)

    t = tautomers[1]
    t.perform_tautomer_transformation()
    assert(neutromeratio.utils.get_stereotag_of_stereobonds(t.initial_state_mol) == 'STEREOE')


    # test if t1 centric mapping of hydrogen, heavy atom acceptor and donor works
    assert(t.initial_state_ligand_atoms == 'CCOCCCOHHHHHHHH')
    assert(t.hybrid_hydrogen_idx_at_lambda_0 == 14)
    assert(t.heavy_atom_hydrogen_acceptor_idx == 3)
    assert(t.heavy_atom_hydrogen_donor_idx == 6)

    # test if dual topology hybrid works
    assert(t.hybrid_atoms == 'CCOCCCOHHHHHHHHH')
    assert(t.hybrid_hydrogen_idx_at_lambda_0 == 14)
    assert(t.hybrid_hydrogen_idx_at_lambda_1 == 15)

    ###################################
    ###################################

    name = 'molDWRow_37'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles)
    assert(len(tautomers) == 2)
    t = tautomers[0]
    t.perform_tautomer_transformation()
    assert(neutromeratio.utils.get_stereotag_of_stereobonds(t.initial_state_mol) == 'STEREOZ')

    # test if t1 centric mapping of hydrogen, heavy atom acceptor and donor works
    assert(t.initial_state_ligand_atoms == 'CCOCCCCCOHHHHHHHHHH')
    assert(t.hydrogen_idx == 12)
    assert(t.heavy_atom_hydrogen_acceptor_idx == 8)
    assert(t.heavy_atom_hydrogen_donor_idx == 2)

    # test if dual topology hybrid works
    assert(t.hybrid_atoms == 'CCOCCCCCOHHHHHHHHHHH')
    assert(t.hybrid_hydrogen_idx_at_lambda_0 == 12)
    assert(t.hybrid_hydrogen_idx_at_lambda_1 == 19)

    t = tautomers[1]
    t.perform_tautomer_transformation()
    assert(neutromeratio.utils.get_stereotag_of_stereobonds(t.initial_state_mol) == 'STEREOE')

    # test if t1 centric mapping of hydrogen, heavy atom acceptor and donor works
    assert(t.initial_state_ligand_atoms == 'CCOCCCCCOHHHHHHHHHH')
    assert(t.hydrogen_idx == 12)
    assert(t.heavy_atom_hydrogen_acceptor_idx == 8)
    assert(t.heavy_atom_hydrogen_donor_idx == 2)

    # test if dual topology hybrid works
    assert(t.hybrid_atoms == 'CCOCCCCCOHHHHHHHHHHH')
    assert(t.hybrid_hydrogen_idx_at_lambda_0 == 12)
    assert(t.hybrid_hydrogen_idx_at_lambda_1 == 19)

    # test if droplet works
    t.add_droplet(t.final_state_ligand_topology, t.get_final_state_ligand_coords(0))


    name = 'molDWRow_1233'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles)
    assert(len(tautomers) == 2)
    t = tautomers[0]
    t.perform_tautomer_transformation()
    assert(neutromeratio.utils.get_stereotag_of_stereobonds(t.initial_state_mol) == 'STEREOZ')

    # test if t1 centric mapping of hydrogen, heavy atom acceptor and donor works
    assert(t.initial_state_ligand_atoms == 'NCOCCCCCCCNNHHHHHHH')
    assert(t.hydrogen_idx == 18)
    assert(t.heavy_atom_hydrogen_acceptor_idx == 0)
    assert(t.heavy_atom_hydrogen_donor_idx == 11)

    # test if dual topology hybrid works
    assert(t.hybrid_atoms == 'NCOCCCCCCCNNHHHHHHHH')
    assert(t.hybrid_hydrogen_idx_at_lambda_0 == 18)
    assert(t.hybrid_hydrogen_idx_at_lambda_1 == 19)

    t = tautomers[1]
    t.perform_tautomer_transformation()
    assert(neutromeratio.utils.get_stereotag_of_stereobonds(t.initial_state_mol) == 'STEREOE')

    # test if t1 centric mapping of hydrogen, heavy atom acceptor and donor works
    assert(t.initial_state_ligand_atoms == 'NCOCCCCCCCNNHHHHHHH')
    assert(t.hydrogen_idx == 18)
    assert(t.heavy_atom_hydrogen_acceptor_idx == 0)
    assert(t.heavy_atom_hydrogen_donor_idx == 11)

    # test if dual topology hybrid works
    assert(t.hybrid_atoms == 'NCOCCCCCCCNNHHHHHHHH')
    assert(t.hybrid_hydrogen_idx_at_lambda_0 == 18)
    assert(t.hybrid_hydrogen_idx_at_lambda_1 == 19)

    # test if droplet works
    t.add_droplet(t.final_state_ligand_topology, t.get_final_state_ligand_coords(0))

def test_tautomer_transformation_for_all_systems():
    from neutromeratio.tautomers import Tautomer
    from ..constants import  exclude_set_ANI, mols_with_charge, multiple_stereobonds
    from ..constants import _get_names
    import random
   
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    name_list = _get_names()
    random.shuffle(name_list)
    ###################################
    ###################################
    for name in name_list[:10]:
        if name in exclude_set_ANI + mols_with_charge + multiple_stereobonds:
            continue
        t1_smiles = exp_results[name]['t1-smiles']
        t2_smiles = exp_results[name]['t2-smiles']

        t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles)
        t = tautomers[0]
        t.perform_tautomer_transformation()


def test_species_conversion():
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x, ANI1x, ANI2x, ANI1ccx
    import random, shutil
    import parmed as pm
    import numpy as np
    from ..constants import _get_names

    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    name_list = _get_names()
    ###################################
    ###################################
    name =  name_list[10]
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles)
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()
    
    for model in [ANI1x, ANI2x, ANI1ccx]:
        m = model()
        energy_function = neutromeratio.ANI1_force_and_energy(
            model=m,
            atoms=tautomer.initial_state_ligand_atoms,
            mol=None)
        
        print(m.species_to_tensor(tautomer.initial_state_ligand_atoms))
    
    for model in [AlchemicalANI2x, AlchemicalANI1ccx]:
        m = model([1,2])
        energy_function = neutromeratio.ANI1_force_and_energy(
            model=m,
            atoms=tautomer.initial_state_ligand_atoms,
            mol=None)
        
        print(m.species_to_tensor(tautomer.initial_state_ligand_atoms))
    
def test_tochani_neutromeratio_sync():

    import torch
    import torchani
    from ..ani import ANI1ccx, ANI1x, ANI2x

    device = torch.device('cpu')
    model_torchani_ANI2x = torchani.models.ANI2x(periodic_table_index=True).to(device)
    model_torchani_ANI1ccx = torchani.models.ANI1ccx(periodic_table_index=True).to(device)
    model_torchani_ANIx = torchani.models.ANI1x(periodic_table_index=True).to(device)

    model_neutromeratio_ANI1ccx = ANI1ccx(periodic_table_index=True).to(device)
    model_neutromeratio_ANI1x = ANI1x(periodic_table_index=True).to(device)
    model_neutromeratio_ANI2x = ANI2x(periodic_table_index=True).to(device)


    coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                                [-0.83140486, 0.39370209, -0.26395324],
                                [-0.66518241, -0.84461308, 0.20759389],
                                [0.45554739, 0.54289633, 0.81170881],
                                [0.66091919, -0.16799635, -0.91037834]]],
                            requires_grad=True, device=device)
    # In periodic table, C = 6 and H = 1
    species = torch.tensor([[6, 1, 1, 1, 1]], device=device)

    for model_torchani, model_neutromeratio in zip(
        [model_torchani_ANIx, model_torchani_ANI1ccx, model_torchani_ANI2x],
        [model_neutromeratio_ANI1x, model_neutromeratio_ANI1ccx, model_neutromeratio_ANI2x],
            ):
        energy_torchani = model_torchani((species, coordinates)).energies
        e1 = energy_torchani.item()
        print('Energy:', energy_torchani.item())

        energy_neutromeratio = model_neutromeratio((species, coordinates, True)).energies
        print('Energy:', energy_neutromeratio.item())
        e2 = energy_neutromeratio.item()
        assert(e1 == e2)


def test_setup_tautomer_system_in_vaccum():
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x
    from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds
    import random, shutil
    import parmed as pm
    from ..constants import _get_names

    names = _get_names()
    lambda_value = 0.1
    random.shuffle(names)
    for name in names[:10]:
        energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
            name=name,
            env='vacuum',
            ANImodel=AlchemicalANI1ccx)
        x0 = tautomer.get_hybrid_coordinates()
        f = energy_function.calculate_force(x0, lambda_value)
        f = energy_function.calculate_energy(x0, lambda_value)


    lambda_value = 0.1
    random.shuffle(names)
    for name in names[:10]:
        energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
            name=name,
            env='vacuum',
            ANImodel=AlchemicalANI2x)
        x0 = tautomer.get_hybrid_coordinates()
        f = energy_function.calculate_force(x0, lambda_value)
        f = energy_function.calculate_energy(x0, lambda_value)


def test_setup_tautomer_system_in_droplet():
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x
    # NOTE: Sometimes this test fails? something wrong with molDWRow_68?
    from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds
    import random, shutil
    from ..constants import _get_names

    names = _get_names()
    lambda_value = 0.1
    random.shuffle(names)
    try:
        for name in names[:10]:
            print(name)
            energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
                name=name,
                env='droplet',
                ANImodel=AlchemicalANI1ccx,
                base_path='pdbs-ani1ccx',
                diameter=10)
            x0 = tautomer.get_ligand_in_water_coordinates()
            energy_function.calculate_force(x0, lambda_value)
            energy_function.calculate_energy(x0, lambda_value)

    finally:
        shutil.rmtree('pdbs-ani1ccx')

    lambda_value = 0.1
    random.shuffle(names)
    try:
        for name in names[:10]:
            print(name)
            energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
                name=name,
                env='droplet',
                ANImodel=AlchemicalANI2x,
                base_path='pdbs-ani2x',
                diameter=10)
            x0 = tautomer.get_ligand_in_water_coordinates()
            energy_function.calculate_force(x0, lambda_value)
            energy_function.calculate_energy(x0, lambda_value)

    finally:
        shutil.rmtree('pdbs-ani2x')

def test_setup_tautomer_system_in_droplet_with_pdbs():
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds
    import random, shutil
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x
    from ..constants import _get_names

    names = _get_names()
    lambda_value = 0.0
    for name in names[:10]:
        print(name)
        energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
            name=name,
            env='droplet',
            ANImodel=AlchemicalANI2x,
            base_path=f'data/droplet_test/{name}',
            diameter=16)
        x0 = tautomer.get_ligand_in_water_coordinates()
        energy_function.calculate_force(x0, lambda_value)


def _get_traj(traj_path, top_path, remove_idx = None):
    top = md.load(top_path).topology
    traj = md.load(traj_path, top=top)
    atoms = [a for a in range(top.n_atoms)]
    if remove_idx:
        print(atoms)
        atoms.remove(remove_idx)
        print(atoms)
        traj = traj.atom_slice(atoms)
    return traj, top

def test_neutromeratio_energy_calculations_with_torchANI_model():

    from ..tautomers import Tautomer
    from ..constants import kT
    from ..analysis import setup_alchemical_system_and_energy_function
    import numpy as np
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x

    # read in exp_results.pickle
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    # vacuum system
    # generate smiles
    name = 'molDWRow_298'
    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name,
        ANImodel=AlchemicalANI1ccx,
        env='vacuum')

    # read in pregenerated traj
    traj_path = 'data/vacuum/molDWRow_298/molDWRow_298_lambda_0.0000_in_vacuum.dcd'
    top_path = 'data/vacuum/molDWRow_298/molDWRow_298.pdb'
    traj, top = _get_traj(traj_path, top_path, tautomer.hybrid_hydrogen_idx_at_lambda_0)
    coordinates = [x.xyz[0] for x in traj[:10]] * unit.nanometer
    
    model = neutromeratio.ani.ANI1ccx()
    torch.set_num_threads(1)
    energy_function = neutromeratio.ANI1_force_and_energy(
        model=model,
        atoms=tautomer.initial_state_ligand_atoms,
        mol=None)
    energy = energy_function.calculate_energy(coordinates)

    assert(is_quantity_close(energy.energy[0].in_units_of(unit.kilojoule_per_mole), (-906555.29945346 * unit.kilojoule_per_mole), rtol=1e-5))

    model = neutromeratio.ani.ANI2x()
    torch.set_num_threads(1)
    energy_function = neutromeratio.ANI1_force_and_energy(
        model=model,
        atoms=tautomer.initial_state_ligand_atoms,
        mol=None)
    energy = energy_function.calculate_energy(coordinates)

    assert(is_quantity_close(energy.energy[0].in_units_of(unit.kilojoule_per_mole), (-907243.8987177598 * unit.kilojoule_per_mole), rtol=1e-5))

    # droplet system
    name = 'molDWRow_298'
    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(name=name, ANImodel=AlchemicalANI1ccx, env='droplet', diameter=16, base_path='data/droplet/molDWRow_298/')

    # read in pregenerated traj
    traj_path = 'data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd'
    top_path = 'data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb'
    traj, top = _get_traj(traj_path, top_path, tautomer.hybrid_hydrogen_idx_at_lambda_0)
    # overwrite the coordinates that rdkit generated with the first frame in the traj
    torch.set_num_threads(1)
    coordinates = [x.xyz[0] for x in traj[:10]] * unit.nanometer

    #ANI1ccx
    model = neutromeratio.ani.ANI1ccx()
    atoms = tautomer.ligand_in_water_atoms[:tautomer.hybrid_hydrogen_idx_at_lambda_0] + tautomer.ligand_in_water_atoms[tautomer.hybrid_hydrogen_idx_at_lambda_0 +1:]
    assert ( len(tautomer.ligand_in_water_atoms) == len(atoms) +1)
    energy_function = neutromeratio.ANI1_force_and_energy(
        model=model,
        atoms=atoms,
        mol=None)
    energy = energy_function.calculate_energy(coordinates)

    assert(is_quantity_close(energy.energy[0].in_units_of(unit.kilojoule_per_mole), (-23572811.32262513 * unit.kilojoule_per_mole), rtol=1e-5))

    #ANI2x
    model = neutromeratio.ani.ANI2x()
    atoms = tautomer.ligand_in_water_atoms[:tautomer.hybrid_hydrogen_idx_at_lambda_0] + tautomer.ligand_in_water_atoms[tautomer.hybrid_hydrogen_idx_at_lambda_0 +1:]
    assert ( len(tautomer.ligand_in_water_atoms) == len(atoms) +1)
    energy_function = neutromeratio.ANI1_force_and_energy(
        model=model,
        atoms=atoms,
        mol=None)
    energy = energy_function.calculate_energy(coordinates)

    assert(is_quantity_close(energy.energy[0].in_units_of(unit.kilojoule_per_mole), (-23577418.068709385 * unit.kilojoule_per_mole), rtol=1e-5))



def test_neutromeratio_energy_calculations_LinearAlchemicalSingleTopologyANI_model():
    from ..tautomers import Tautomer
    import numpy as np
    from ..constants import kT
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x, ANI1ccx

    # read in exp_results.pickle
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)
    
    ######################################################################
    #vacuum
    ######################################################################
    name = 'molDWRow_298'
    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name,
        env='vacuum',
        ANImodel=AlchemicalANI1ccx)
    # read in pregenerated traj
    traj_path = 'data/vacuum/molDWRow_298/molDWRow_298_lambda_0.0000_in_vacuum.dcd'
    top_path = 'data/vacuum/molDWRow_298/molDWRow_298.pdb'
    traj, top = _get_traj(traj_path, top_path, None)

    x0 = [x.xyz[0] for x in traj[:10]] * unit.nanometer

    energy_1 = energy_function.calculate_energy(x0, lambda_value=1.0)
    assert(is_quantity_close(energy_1.energy[0].in_units_of(unit.kilojoule_per_mole), (-906555.29945346 * unit.kilojoule_per_mole), rtol=1e-9))
    for e1, e2 in zip(energy_1.energy, [-906555.29945346, -905750.20471091, -906317.24952004, -906545.17543265, -906581.65215098, -906618.2832786,  -906565.05631782, -905981.82167316, -904681.20632002, -904296.8214631] * unit.kilojoule_per_mole):
        assert(is_quantity_close(e1, e2, rtol=1e-2))    

     
    energy_0 = energy_function.calculate_energy(x0, lambda_value=0.0)
    assert(is_quantity_close(energy_0.energy[0].in_units_of(unit.kilojoule_per_mole), (-906912.01647632 * unit.kilojoule_per_mole), rtol=1e-9))
    ######################################################################
    # compare with ANI1ccx 
    traj, top = _get_traj(traj_path, top_path, tautomer.hybrid_hydrogen_idx_at_lambda_1)
    model = ANI1ccx()

    energy_function = neutromeratio.ANI1_force_and_energy(
        model=model,
        atoms=tautomer.final_state_ligand_atoms,
        mol=None)

    coordinates= [x.xyz[0] for x in traj[0]] * unit.nanometer
    assert(len(tautomer.initial_state_ligand_atoms) == len(coordinates[0]))
    assert(is_quantity_close(energy_0.energy[0] ,energy_function.calculate_energy(coordinates).energy))

    traj, top = _get_traj(traj_path, top_path, tautomer.hybrid_hydrogen_idx_at_lambda_0)


    energy_function = neutromeratio.ANI1_force_and_energy(
        model=model,
        atoms=tautomer.initial_state_ligand_atoms,
        mol=None)
    
    coordinates= [x.xyz[0] for x in traj[0]] * unit.nanometer
    assert(len(tautomer.final_state_ligand_atoms) == len(coordinates[0]))
    assert(is_quantity_close(energy_1.energy[0] ,energy_function.calculate_energy(coordinates).energy))
    

    ######################################################################
    # droplet
    ######################################################################
    name = 'molDWRow_298'
    energy_function, tautomer, flipped=setup_alchemical_system_and_energy_function(
        name=name,
        ANImodel=AlchemicalANI1ccx,
        env='droplet',
        base_path='data/droplet/molDWRow_298/',
        diameter=18)

    # read in pregenerated traj
    traj_path = 'data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd'
    top_path = 'data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb'
    traj, top = _get_traj(traj_path, top_path, None)

    # overwrite the coordinates that rdkit generated with the first frame in the traj
    x0 = [x.xyz[0] for x in traj[:10]] * unit.nanometer
    energy_1 = energy_function.calculate_energy(x0, lambda_value=1.0)
    energy_0 = energy_function.calculate_energy(x0, lambda_value=0.0)

    assert (len(tautomer.ligand_in_water_atoms) == len(x0[0]))
    print(energy_1.energy.in_units_of(unit.kilojoule_per_mole))

    assert(is_quantity_close(energy_1.energy[0].in_units_of(unit.kilojoule_per_mole), (-23572807.12255169 * unit.kilojoule_per_mole)))
    for e1, e2 in zip(energy_1.energy, [-23572807.122551697, -23572337.15972069, -23572169.49734318, -23572193.66573638, -23572027.39844025, -23571995.93757311, -23571816.9790378, -23571836.9212332, -23571824.36825795, -23571877.94526987] * unit.kilojoule_per_mole):
         assert(is_quantity_close(e1, e2))    

     
    assert(is_quantity_close(energy_0.energy[0].in_units_of(unit.kilojoule_per_mole), (-23573435.102149136 * unit.kilojoule_per_mole)))
    ######################################################################
    # compare with ANI1ccx -- test1
    name = 'molDWRow_298'
    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name,
        ANImodel=AlchemicalANI1ccx,
        env='droplet',
        base_path='data/droplet/molDWRow_298/',
        diameter=18)
    # read in pregenerated traj
    traj_path = 'data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd'
    top_path = 'data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb'
    traj, top=_get_traj(traj_path, top_path, None)
    # remove restraints
    energy_function.list_of_lambda_restraints = []

    # overwrite the coordinates that rdkit generated with the first frame in the traj
    x0 = [x.xyz[0] for x in traj[:10]] * unit.nanometer
    energy_1 = energy_function.calculate_energy(x0, lambda_value=1.0)
    energy_0 = energy_function.calculate_energy(x0, lambda_value=0.0)

    model = ANI1ccx()

    traj, top = _get_traj(traj_path, top_path, tautomer.hybrid_hydrogen_idx_at_lambda_1)
    atoms = tautomer.ligand_in_water_atoms[:tautomer.hybrid_hydrogen_idx_at_lambda_1] + tautomer.ligand_in_water_atoms[tautomer.hybrid_hydrogen_idx_at_lambda_1 +1:]

    energy_function = neutromeratio.ANI1_force_and_energy(
        model=model,
        atoms=atoms,
        mol=None)
    
    coordinates = [x.xyz[0] for x in traj[:10]] * unit.nanometer
    assert (len(atoms) == len(coordinates[0]))
    
    energies_ani1ccx_0 = energy_function.calculate_energy(coordinates)
    assert(is_quantity_close(energy_0.energy[0].in_units_of(unit.kilojoule_per_mole), energies_ani1ccx_0.energy[0].in_units_of(unit.kilojoule_per_mole)))


    ######################################################################
    # compare with ANI1ccx -- test2
    name = 'molDWRow_298'
    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name,
        ANImodel=AlchemicalANI1ccx,
        env='droplet',
        base_path='data/droplet/molDWRow_298/',
        diameter=18
        )

    # read in pregenerated traj
    traj_path = 'data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd'
    top_path = 'data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb'
    traj, top=_get_traj(traj_path, top_path, None)

    x0 = [x.xyz[0] for x in traj[:10]] * unit.nanometer
    energy_1 = energy_function.calculate_energy(x0, lambda_value=1.0)
    energy_0 = energy_function.calculate_energy(x0, lambda_value=0.0)


    model = ANI1ccx()
    traj, top = _get_traj(traj_path, top_path, tautomer.hybrid_hydrogen_idx_at_lambda_1)
    atoms = tautomer.ligand_in_water_atoms[:tautomer.hybrid_hydrogen_idx_at_lambda_1] + tautomer.ligand_in_water_atoms[tautomer.hybrid_hydrogen_idx_at_lambda_1 +1:]

    energy_function = neutromeratio.ANI1_force_and_energy(
        model=model,
        atoms=atoms,
        mol=None)
    
    coordinates = [x.xyz[0] for x in traj[:10]] * unit.nanometer
    assert (len(atoms) == len(coordinates[0]))
    
    energies_ani1ccx_0 = energy_function.calculate_energy(coordinates)

    #subtracting restraint energies
    energy_0_minus_restraint = (energy_0.energy[0] - energy_0.restraint_energy_contribution[0]).in_units_of(unit.kilojoule_per_mole)
    assert(is_quantity_close(energy_0_minus_restraint, energies_ani1ccx_0.energy[0].in_units_of(unit.kilojoule_per_mole)))


def test_restraint():
    from neutromeratio.tautomers import Tautomer
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    name = 'molDWRow_298'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(
        name,
        t1_smiles,
        t2_smiles)
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    atoms = tautomer.initial_state_ligand_atoms
    harmonic = neutromeratio.restraints.BondHarmonicRestraint(
        sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms)
    flat_bottom = neutromeratio.restraints.BondFlatBottomRestraint(
        sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms)

    x0 = tautomer.get_initial_state_ligand_coords(0)
    coordinates = torch.tensor(x0.value_in_unit(unit.nanometer),
                               requires_grad=True, device=device, dtype=torch.float32)

    print('Restraing: {}.'.format(harmonic.restraint(coordinates)))
    print('Restraing: {}.'.format(flat_bottom.restraint(coordinates)))


def test_restraint_with_LinearAlchemicalSingleTopologyANI():
    import numpy as np
    from ..constants import kT
    from ..ani import AlchemicalANI1ccx
    # read in exp_results.pickle
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    # generate smiles
    name = 'molDWRow_298'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(
        name,
        t1_smiles,
        t2_smiles)
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    # read in pregenerated traj
    traj_path = 'data/vacuum/molDWRow_298/molDWRow_298_lambda_0.0000_in_vacuum.dcd'
    top_path = 'data/vacuum/molDWRow_298/molDWRow_298.pdb'
    traj, top = _get_traj(traj_path, top_path, None)
    x0 = [x.xyz[0] for x in traj[0]] * unit.nanometer

    # the first of the alchemical_atoms will be dummy at lambda 0, the second at lambda 1
    # protocoll goes from 0 to 1
    dummy_atoms = [tautomer.hybrid_hydrogen_idx_at_lambda_1, tautomer.hybrid_hydrogen_idx_at_lambda_0]
    atoms = tautomer.hybrid_atoms

    model = AlchemicalANI1ccx(alchemical_atoms=dummy_atoms)

    energy_function = neutromeratio.ANI1_force_and_energy(
        model=model,
        atoms=atoms,
        mol=None)

    energy_function.list_of_restraints = tautomer.ligand_restraints

    energy = energy_function.calculate_energy(x0,lambda_value=0.0)
    assert (is_quantity_close(energy.energy.in_units_of(unit.kilojoule_per_mole), (-906912.01647632 * unit.kilojoule_per_mole), rtol=1e-9))


def test_min_and_single_point_energy():

    from ..ani import ANI1ccx
    # name of the system
    name = 'molDWRow_298'

    # extract smiles
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    nr_of_confs = 10
    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(
        name=name,
        t1_smiles=t1_smiles,
        t2_smiles=t2_smiles,
        nr_of_conformations=nr_of_confs)
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    # set model
    model = ANI1ccx()
    model = model.to(device)
    torch.set_num_threads(1)

    # calculate energy using both structures and pure ANI1ccx
    for ase_mol, ligand_atoms, ligand_coords in zip([tautomer.initial_state_ase_mol, tautomer.final_state_ase_mol],
                                                    [tautomer.initial_state_ligand_atoms, tautomer.final_state_ligand_atoms],
                                                    [tautomer.get_initial_state_ligand_coords, tautomer.get_final_state_ligand_coords]):
        energy_function = neutromeratio.ANI1_force_and_energy(
            model=model,
            atoms=ligand_atoms,
            mol=ase_mol)

        for i in range(nr_of_confs):
            # minimize
            x0, hist_e = energy_function.minimize(ligand_coords(i))
            print(energy_function.calculate_energy(x0).energy)


def test_thermochemistry():
    from ..ani import ANI1ccx
    # name of the system
    name = 'molDWRow_298'

    # extract smiles
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    nr_of_confs = 10
    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(
        name=name,
        t1_smiles=t1_smiles,
        t2_smiles=t2_smiles,
        nr_of_conformations=nr_of_confs)
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    # set model
    model = ANI1ccx()
    model = model.to(device)
    torch.set_num_threads(1)

    # calculate energy using both structures and pure ANI1ccx
    for ase_mol, ligand_atoms, ligand_coords in zip([tautomer.initial_state_ase_mol, tautomer.final_state_ase_mol],
                                                    [tautomer.initial_state_ligand_atoms, tautomer.final_state_ligand_atoms],
                                                    [tautomer.get_initial_state_ligand_coords, tautomer.get_final_state_ligand_coords]):
        energy_function = neutromeratio.ANI1_force_and_energy(
            model=model,
            atoms=ligand_atoms,
            mol=ase_mol,
        )
        for i in range(nr_of_confs):
            # minimize
            x0, hist_e = energy_function.minimize(ligand_coords(i))
            print(x0.shape)
            energy_function.get_thermo_correction(x0) # x has [1][K][3] dimenstion -- N: number of mols, K: number of atoms


def test_equilibrium():
    # test the langevin dynamics with different neural net potentials 
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..equilibrium import LangevinDynamics
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x
    from ..constants import _get_names
    
    # name of the system
    name = 'molDWRow_298'
    # number of steps
    n_steps = 100

    for model_name, model in zip(['AlchemicalANI2x', 'AlchemicalANI1ccx'], [AlchemicalANI2x, AlchemicalANI1ccx]):
        energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
            name=name,
            ANImodel=model,
            env='vacuum',
            base_path='pdbs')

        
        x0 = tautomer.get_hybrid_coordinates() # format [1][K][3] * unit
        x0, hist_e = energy_function.minimize(x0)

        # lambda=1.0
        energy_and_force = lambda x : energy_function.calculate_force(x, 1.0)

        langevin = LangevinDynamics(
            atoms=tautomer.hybrid_atoms,
            energy_and_force=energy_and_force,
            )

        equilibrium_samples, energies, restraint_contribution = langevin.run_dynamics(
            x0,
            n_steps=n_steps,
            stepsize=1.0 * unit.femtosecond,
            progress_bar=True)

        equilibrium_samples = [x[0].value_in_unit(unit.nanometer) for x in equilibrium_samples]
        traj = md.Trajectory(equilibrium_samples, tautomer.hybrid_topology)
        traj.save(f'test_{model_name}.dcd', force_overwrite=True)
        traj[0].save('test.pdb')

        # lambda=0.0
        energy_and_force = lambda x : energy_function.calculate_force(x, 0.0)

        langevin = LangevinDynamics(
            atoms=tautomer.hybrid_atoms,
            energy_and_force=energy_and_force
            )

        equilibrium_samples, energies, restraint_contribution = langevin.run_dynamics(
            x0,
            n_steps=n_steps,
            stepsize=1.0*unit.femtosecond,
            progress_bar=True)

def test_setup_energy_function():
    # test the seupup of the energy function with different alchemical potentials
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI2x, ANI1ccx
    name = 'molDWRow_298'

    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name,
        env='vacuum',
        ANImodel=AlchemicalANI2x)
    assert (flipped == True)
    
    failed = False
    try:
        # this should to fail
        energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name,
        env='vacuum',
        ANImodel=ANI1ccx)
    except RuntimeError:
        failed = True
        pass

    # make sure that setup_alchemical_system_and_energy_function has failed with non-alchemical potential
    assert(failed == True)



def test_memory_issue():
    # test the seup of the energy function with different alchemical potentials
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI2x, ANI1ccx
    from glob import glob
    from ..constants import device
    import torchani, torch

    # either use neutromeratio ANI object
    nn = ANI1ccx(periodic_table_index=True).to(device)
    # or torchani ANI object
    nn = torchani.models.ANI1ccx(periodic_table_index=True).to(device)
   
    # read in trajs
    name = 'molDWRow_298'
    data_path = 'data/droplet/'
    max_snapshots_per_window = 10
    dcds = glob(f"{data_path}/{name}/*.dcd")

    md_trajs = []
    energies = []

    # read in all the frames from the trajectories
    top = f"{data_path}/{name}/{name}_in_droplet.pdb"

    species = []
    for dcd_filename in dcds:
        traj = md.load_dcd(dcd_filename, top=top)
        snapshots = traj.xyz * unit.nanometer
        further_thinning = max(int(len(snapshots) / max_snapshots_per_window), 1)
        snapshots = snapshots[::further_thinning][:max_snapshots_per_window]
        coordinates = [sample / unit.angstrom for sample in snapshots] * unit.angstrom
        md_trajs.append(coordinates)

    # generate species
    for a in traj.topology.atoms:
        species.append(a.element.symbol)

    element_index = { 'C' : 6, 'N' : 7, 'O' : 8, 'H' : 1}
    species = [element_index[e] for e in species]
    
    # calcualte energies
    for traj in md_trajs:
        for xyz in traj:
            coordinates = torch.tensor([xyz.value_in_unit(unit.nanometer)],
                                requires_grad=False, device=device, dtype=torch.float32)

            species_tensor = torch.tensor([species] * len(coordinates), device=device, requires_grad=False)
            energy = nn((species_tensor, coordinates)).energies
            print(energy)
        energies.append(energy)


def calculate_single_energy():
    import torch
    import torchani

    torch.set_num_threads(1)
    n_snapshots, n_atoms = 1, 50

    device = torch.device('cpu')
    model = torchani.models.ANI1ccx(periodic_table_index=True).to(device)

    element_index = {'C': 6, 'N': 7, 'O': 8, 'H': 1}
    species = [element_index[e] for e in 'CH' * n_atoms][:n_atoms]
    species_tensor = torch.tensor([species] * n_snapshots, device=device)


    # condition 1: computing n_snapshots energies
    print('computing once')
    coordinates = torch.tensor(torch.randn((n_snapshots, n_atoms, 3)),
                               requires_grad=True, device=device, dtype=torch.float32)
    energy = model((species_tensor, coordinates)).energies

    print(energy)


def test_setup_mbar():
    # test the setup mbar function with different models, environments and potentials
    from ..parameter_gradients import setup_mbar
    from ..ani import AlchemicalANI2x, AlchemicalANI1x, AlchemicalANI1ccx


    name = 'molDWRow_298'

    # vacuum
    fec = setup_mbar(
        name,
        env='vacuum',
        data_path="data/vacuum",
        ANImodel=AlchemicalANI1ccx,
        bulk_energy_calculation=False,
        max_snapshots_per_window=100)
    assert(np.isclose(-0.2159245230162501, fec.end_state_free_energy_difference[0]))

    fec = setup_mbar(
        name,
        env='vacuum',
        data_path="data/vacuum",
        ANImodel=AlchemicalANI2x,
        bulk_energy_calculation=False,
        max_snapshots_per_window=100)
    assert(np.isclose(-7.981541822664637, fec.end_state_free_energy_difference[0]))

    fec = setup_mbar(
        name,
        env='vacuum',
        data_path="data/vacuum",
        ANImodel=AlchemicalANI1x,
        bulk_energy_calculation=False,
        max_snapshots_per_window=100)
    assert(np.isclose(-8.916705062503821, fec.end_state_free_energy_difference[0]))

    # droplet
    fec = setup_mbar(
        name,
        env='droplet',
        diameter=18,
        data_path="data/droplet",
        ANImodel=AlchemicalANI1ccx,
        bulk_energy_calculation=False,
        max_snapshots_per_window=50)
    assert(np.isclose(-2.943846274310164, fec.end_state_free_energy_difference[0]))

    fec = setup_mbar(
        name,
        env='droplet',
        diameter=18,
        data_path="data/droplet",
        ANImodel=AlchemicalANI2x,
        bulk_energy_calculation=False,
        max_snapshots_per_window=50)
    assert(np.isclose(-15.878144843280154, fec.end_state_free_energy_difference[0]))

    fec = setup_mbar(
        name,
        env='droplet',
        diameter=18,
        data_path="data/droplet",
        ANImodel=AlchemicalANI1x,
        bulk_energy_calculation=False,
        max_snapshots_per_window=50)
    assert(np.isclose(-11.071049647338729, fec.end_state_free_energy_difference[0]))

def test_change_stereobond():
    from ..utils import change_only_stereobond, get_nr_of_stereobonds
    def get_all_stereotags(smiles):
        mol = Chem.MolFromSmiles(smiles)
        stereo = set()
        for bond in mol.GetBonds():
            if str(bond.GetStereo()) == 'STEREONONE':
                continue
            else:
                stereo.add(bond.GetStereo())
        return stereo
    
    name = 'molDWRow_298'
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
    
    # no stereobond
    smiles = exp_results[name]['t1-smiles']
    assert(get_nr_of_stereobonds(smiles) == 0)
    stereo1 = get_all_stereotags(smiles)
    stereo2 = get_all_stereotags(change_only_stereobond(smiles))
    assert(stereo1 == stereo2)

    # one stereobond
    smiles = exp_results[name]['t2-smiles']
    assert(get_nr_of_stereobonds(smiles) == 1)
    stereo1 = get_all_stereotags(smiles)
    stereo2 = get_all_stereotags(change_only_stereobond(smiles))
    assert(stereo1 != stereo2)


def test_tautomer_conformation():
    from ..ani import ANI1ccx
    # name of the system
    name = 'molDWRow_298'
    # number of steps

    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))

    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(
        name,
        t1_smiles,
        t2_smiles,
        nr_of_conformations=5)
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    print(f"Nr of initial conformations: {tautomer.get_nr_of_initial_state_ligand_coords()}")
    print(f"Nr of final conformations: {tautomer.get_nr_of_final_state_ligand_coords()}")

    assert(tautomer.get_nr_of_initial_state_ligand_coords() == 5)
    assert(tautomer.get_nr_of_final_state_ligand_coords() == 5)

    # set model
    model = ANI1ccx()
    model = model.to(device)
    torch.set_num_threads(1)

    # calculate energy using both structures and pure ANI1ccx
    for ase_mol, ligand_atoms, get_ligand_coords in zip([tautomer.initial_state_ase_mol, tautomer.final_state_ase_mol], [tautomer.initial_state_ligand_atoms, tautomer.final_state_ligand_atoms], [tautomer.get_initial_state_ligand_coords, tautomer.get_final_state_ligand_coords]):
        energy_function = neutromeratio.ANI1_force_and_energy(
            model=model,
            atoms=ligand_atoms,
            mol=ase_mol,
        )

        for conf_id in range(5):
            # minimize
            print(f"Conf: {conf_id}")
            x, e_min_history = energy_function.minimize(get_ligand_coords(conf_id), maxiter=100000)
            energy = energy_function.calculate_energy(x) # coordinates need to be in [N][K][3] format
            e_correction = energy_function.get_thermo_correction(x)
            print(f"Energy: {energy.energy}")
            print(f"Energy correction: {e_correction}")


def test_mining_minima():
    # name of the system
    name = 'molDWRow_298'
    # number of steps
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
    torch.set_num_threads(1)

    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(
        name,
        t1_smiles,
        t2_smiles)
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    confs_traj, e, minimum_energies, all_energies, all_conformations = tautomer.generate_mining_minima_structures()


def test_plotting():

    from neutromeratio.constants import kT
    from neutromeratio.plotting import plot_correlation_analysis
    results = pickle.load(open('neutromeratio/data/all_results.pickle', 'rb'))

    x_list = []
    y_list = []

    for a in list(results.ddG_DFT):
        a = a * kT
        a = a.value_in_unit(unit.kilocalorie_per_mole)
        x_list.append(a)

    for a in list(results.experimental_values):
        a = a * kT
        a = a.value_in_unit(unit.kilocalorie_per_mole)
        y_list.append(a)

    df = pd.DataFrame(list(zip(list(results.names),
                               x_list,
                               y_list,
                               ['B3LYP/aug-cc-pVTZ']*len(results.names))),
                      columns=['names', 'x', 'y', 'method'])

    plot_correlation_analysis(
        df, 'DFT(B3LYP/aug-cc-pVTZ) in vacuum vs experimental data in solution', 'test1', 'test2', 'g', 'o')


def test_generating_droplet():
    from ..constants import kT
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..utils import generate_tautomer_class_stereobond_aware
    from ..ani import AlchemicalANI1ccx
    import numpy as np

    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))

    name = 'molDWRow_298'
    diameter=16
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    # generate both rdkit mol
    t_type, tautomers, flipped = generate_tautomer_class_stereobond_aware(
        name, 
        t1_smiles,
        t2_smiles,
        nr_of_conformations=5)

    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()
    m = tautomer.add_droplet(tautomer.hybrid_topology,
                             tautomer.get_hybrid_coordinates(),
                             diameter=diameter * unit.angstrom,
                             restrain_hydrogen_bonds=True,
                             restrain_hydrogen_angles=False,
                             top_file=f"data/{name}_in_droplet.pdb")

    # define the alchemical atoms
    alchemical_atoms = [tautomer.hybrid_hydrogen_idx_at_lambda_1, tautomer.hybrid_hydrogen_idx_at_lambda_0]

    # extract hydrogen donor idx and hydrogen idx for from_mol
    model = AlchemicalANI1ccx(alchemical_atoms=alchemical_atoms)
    model = model.to(device)

    # perform initial sampling
    energy_function = neutromeratio.ANI1_force_and_energy(
        model=model,
        atoms=tautomer.ligand_in_water_atoms,
        mol=None,
    )

    for r in tautomer.ligand_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)

    for r in tautomer.hybrid_ligand_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)

    x0 = tautomer.get_ligand_in_water_coordinates()
    energy = energy_function.calculate_energy(x0)
    assert(is_quantity_close(energy.energy[0], (-15547479.771537919 * unit.kilojoule_per_mole)))

    energy = energy_function.calculate_energy(x0)
    assert(is_quantity_close(energy.energy[0], (-15547479.771537919 * unit.kilojoule_per_mole)))

    tautomer.add_COM_for_hybrid_ligand(np.array([diameter/2, diameter/2, diameter/2]) * unit.angstrom)

    for r in tautomer.solvent_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)

    for r in tautomer.com_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)

    energy = energy_function.calculate_energy(x0)
    assert(is_quantity_close(energy.energy[0], (-15547319.00691153 * unit.kilojoule_per_mole)))
    
    
    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name,
        env='droplet',
        ANImodel=AlchemicalANI1ccx,
        base_path='data',
        diameter=diameter)

    energy = energy_function.calculate_energy(x0)
    assert(is_quantity_close(energy.energy[0], (-15547319.00691153 * unit.kilojoule_per_mole), rtol=1e-7))

@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Psi4 import fails on travis."
)
def test_psi4():
    from neutromeratio import qmpsi4
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))

    name = 'molDWRow_298'

    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    # generate both rdkit mol
    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles, nr_of_conformations=5)
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    mol = tautomer.initial_state_mol

    psi4_mol = qm.mol2psi4(mol, 1)
    qm.optimize(psi4_mol)

def test_orca_input_generation():
    from neutromeratio import qmorca
    name = 'molDWRow_298'
   
    
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    qm_results = pickle.load(open('data/results/qm_results.pickle', 'rb'))
    mol = qm_results[name][t1_smiles]['vac'][0]
    orca_input = qmorca.generate_orca_script_for_solvation_free_energy(mol, 0)
    print(orca_input)

@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Orca not installed."
)
def test_running_orca():
    from neutromeratio import qmorca
    name = 'molDWRow_298'
   
    
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    qm_results = pickle.load(open('data/results/qm_results.pickle', 'rb'))
    mol = qm_results[name][t1_smiles]['vac'][0]
    orca_input = qmorca.generate_orca_script_for_solvation_free_energy(mol, 0)
    f = open('tmp.inp', 'w+')
    f.write(orca_input)
    f.close()
    out = qmorca.run_orca('tmp.inp')


def test_solvate_orca():
    from neutromeratio import qmorca
    import re
    from simtk import unit
    from ..constants import hartree_to_kJ_mol
    name = 'molDWRow_298'
    
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    qm_results = pickle.load(open('data/results/qm_results.pickle', 'rb'))
    mol = qm_results[name][t1_smiles]['vac'][0]
    orca_input = qmorca.generate_orca_script_for_solvation_free_energy(mol, 0)
    f = open('tmp.inp', 'w+')
    f.write(orca_input)
    f.close()
    rc, output, err = qmorca.run_orca('tmp.inp')
    
    output_str = output.decode('UTF-8')

    try:
        #Total Energy after SMD CDS correction : 
        found = re.search('Total Energy after SMD CDS correction =\s*([-+]?\d*\.\d+|\d+)\s*Eh', output_str).group(1)
    except AttributeError:
        found = '' # apply your error handling

    if not found:
        print(found)
        print(output_str)
        raise RuntimeError('Something not working. Aborting')
    
    
    print(found)
    print(output_str)

    E_in_solvent = (float(found) * hartree_to_kJ_mol) * unit.kilojoule_per_mole
    print(E_in_solvent)
    assert(E_in_solvent == -907364.6683318849 * unit.kilojoule_per_mole)

def test_io_checkpoints():
    from ..parameter_gradients import _save_checkpoint, _load_checkpoint, _get_nn_layers
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x
    
    # specify the system you want to simulate
    for idx, (model, model_name) in enumerate(zip(
        [AlchemicalANI1ccx, AlchemicalANI2x, AlchemicalANI1x],
        ['AlchemicalANI1ccx', 'AlchemicalANI2x', 'AlchemicalANI1x'])):
        # set tweaked parameters
        print(model_name)
        model_instance = model([0,0])
        AdamW, AdamW_scheduler, SGD, SGD_scheduler = _get_nn_layers(8, model_instance)
        # initial parameters
        params1 = list(model.tweaked_neural_network.parameters())[6][0].tolist()
        _load_checkpoint(f'data/{model_name}_3.pt', model_instance, AdamW, AdamW_scheduler, SGD, SGD_scheduler)
        # load parameters
        params2 = list(model_instance.tweaked_neural_network.parameters())[6][0].tolist()
        # make sure somehting happend
        assert (params1 != params2)
        # test that new instances have the new parameters
        m = model([0,0])
        params3 = list(m.tweaked_neural_network.parameters())[6][0].tolist()
        assert (params2 == params3)


def test_load_parameters():
    from ..parameter_gradients import _save_checkpoint, _load_checkpoint, _get_nn_layers
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x
    
    # specify the system you want to simulate
    for idx, (model, model_name) in enumerate(zip(
        [AlchemicalANI1ccx, AlchemicalANI2x, AlchemicalANI1x],
        ['AlchemicalANI1ccx', 'AlchemicalANI2x', 'AlchemicalANI1x'])):
        # set tweaked parameters
        print(model_name)
        model_instance = model([0,0])
        # initial parameters
        params1 = list(model_instance.original_neural_network.parameters())[6][0].tolist()
        params2 = list(model_instance.tweaked_neural_network.parameters())[6][0].tolist()
        assert (params1 == params2)
        # make sure somehting happend
        model_instance.load_nn_parameters(f'data/{model_name}_3.pt', extract_from_checkpoint=True)
        params1 = list(model_instance.original_neural_network.parameters())[6][0].tolist()
        params2 = list(model_instance.tweaked_neural_network.parameters())[6][0].tolist()
        assert (params1 != params2)
        # test that new instances have the new parameters
        m = model([0,0])
        params3 = list(m.tweaked_neural_network.parameters())[6][0].tolist()
        assert (params2 == params3)


def test_parameter_gradient():
    from ..constants import mols_with_charge, exclude_set_ANI, kT, multiple_stereobonds
    from tqdm import tqdm
    from ..parameter_gradients import FreeEnergyCalculator
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x
    
    # nr of steps
    #################
    n_steps = 100
    #################

    # specify the system you want to simulate
    name = 'molDWRow_298'  #Experimental free energy difference: 1.132369 kcal/mol
    for model in [AlchemicalANI1ccx, AlchemicalANI2x, AlchemicalANI1x]:
        
        energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
            name,
            env='vacuum',
            ANImodel=model)
        x0 = tautomer.get_hybrid_coordinates()
        potential_energy_trajs = []
        ani_trajs = []
        lambdas = np.linspace(0, 1, 5)

        for lamb in tqdm(lambdas):
            # minimize coordinates with a given lambda value
            x0, e_history = energy_function.minimize(x0, maxiter=5000, lambda_value=lamb)
            # define energy function with a given lambda value
            energy_and_force = lambda x : energy_function.calculate_force(x, lamb)
            # define langevin object with a given energy function
            langevin = neutromeratio.LangevinDynamics(
                                            atoms=tautomer.hybrid_atoms,
                                            energy_and_force=energy_and_force)

            # sampling
            equilibrium_samples, energies, restraint_energies = langevin.run_dynamics(x0,
                                                                            n_steps=n_steps,
                                                                            stepsize=1.0*unit.femtosecond,
                                                                            progress_bar=False)
            potential_energy_trajs.append(energies)
            
            ani_trajs.append(md.Trajectory([x[0] / unit.nanometer for x in equilibrium_samples], tautomer.hybrid_topology))

        # calculate free energy in kT
        fec = FreeEnergyCalculator(ani_model=energy_function,
                                md_trajs=ani_trajs,
                                potential_energy_trajs=potential_energy_trajs,
                                lambdas=lambdas,
                                bulk_energy_calculation=False,
                                n_atoms=len(tautomer.hybrid_atoms),
                                max_snapshots_per_window=-1)

        # BEWARE HERE: I change the sign of the result since if flipped is TRUE I have 
        # swapped tautomer 1 and 2 to mutate from the tautomer WITH the stereobond to the 
        # one without the stereobond
        if flipped:
            deltaF = fec.compute_free_energy_difference() * -1
        else:
            deltaF = fec.compute_free_energy_difference()
        print(f"Free energy difference {(deltaF.item() * kT).value_in_unit(unit.kilocalorie_per_mole)} kcal/mol")

        deltaF.backward()  # no errors or warnings
        params = list(energy_function.model.tweaked_neural_network.parameters())
        none_counter = 0
        for p in params:
            if(p.grad == None):  # some are None!
                none_counter += 1

        if not (len(params) == 256 or len(params) == 448):
            raise RuntimeError() 
        if not (none_counter == 64 or none_counter == 256):
            raise RuntimeError()



def test_thinning():
    from glob import glob
    dcds = glob(f"data/droplet/molDWRow_298/*dcd")
    top = f"data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    print(dcds)
    for f in dcds:
        print(f)
        traj = md.load_dcd(f, top=top)
        print(len(traj))
    #raise RuntimeError('t')

def test_fec():
    from ..parameter_gradients import get_perturbed_free_energy_differences
    from ..constants import kT, device, exclude_set_ANI, mols_with_charge
    from ..parameter_gradients import setup_mbar
    from glob import glob
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x
    import numpy as np

    env = 'vacuum'
    names = ['molDWRow_298', 'SAMPLmol2']


    for idx, model in enumerate([AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x ]):

        if idx == 0:
            #testing fec
            bulk_energy_calculation=True
            fec_list = [setup_mbar(
                name,
                ANImodel=model,
                env=env,
                bulk_energy_calculation=bulk_energy_calculation,
                data_path='./data/vacuum',
                max_snapshots_per_window=80) for name in names
                ]

            assert(len(fec_list) == 2)
            fec = get_perturbed_free_energy_differences(fec_list)
            print(fec)
            for e1, e2 in zip(fec, [1.2104192392435253, -5.316053972628578]):
                assert(np.isclose(e1.item(),e2, rtol=1e-4)) 

            # testing fec
            bulk_energy_calculation=False
            fec_list = [setup_mbar(
                 name,
                ANImodel=model,
                env=env,
                bulk_energy_calculation=bulk_energy_calculation,
                data_path='./data/vacuum',
                max_snapshots_per_window=80) for name in names
                ]

            assert(len(fec_list) == 2)
            fec = get_perturbed_free_energy_differences(fec_list)
            print(fec)
            for e1, e2 in zip(fec, [1.2104192392435253, -5.316053972628578]):
                assert(np.isclose(e1.item(),e2, rtol=1e-4)) 

            fec = setup_mbar(
                'molDWRow_298',
                ANImodel=model,
                env=env,
                bulk_energy_calculation=True,
                data_path='./data/vacuum',
                max_snapshots_per_window=80)
            assert(np.isclose(fec.end_state_free_energy_difference[0], fec.compute_free_energy_difference().item(), rtol=1e-5))

        if idx == 1:
            bulk_energy_calculation=True
            fec_list = [setup_mbar(
                name,
                ANImodel=model,
                env=env,
                bulk_energy_calculation=bulk_energy_calculation,
                data_path='./data/vacuum',
                max_snapshots_per_window=60) for name in names
                ]

            assert(len(fec_list) == 2)
            fec = get_perturbed_free_energy_differences(fec_list)
            print(fec)
            for e1, e2 in zip(fec, [10.3192, -9.7464]):
                assert(np.isclose(e1.item(),e2, rtol=1e-4)) 

        if idx == 2:
            bulk_energy_calculation=True
            fec_list = [setup_mbar(
                name,
                ANImodel=model,
                env=env,
                bulk_energy_calculation=bulk_energy_calculation,
                data_path='./data/vacuum',
                max_snapshots_per_window=60) for name in names
                ]

            assert(len(fec_list) == 2)
            fec = get_perturbed_free_energy_differences(fec_list)
            print(fec)
            for e1, e2 in zip(fec, [8.8213, -9.6649]):
                assert(np.isclose(e1.item(),e2, rtol=1e-4)) 


def test_ess():
    from ..parameter_gradients import get_perturbed_free_energy_differences
    from ..parameter_gradients import setup_mbar
    from ..ani import AlchemicalANI1ccx
    import numpy as np

    env = 'vacuum'
    name = 'molDWRow_298'


    model = AlchemicalANI1ccx

    fec = setup_mbar(
        name,
        ANImodel=model,
        env=env,
        bulk_energy_calculation=False,
        data_path='./data/vacuum',
        max_snapshots_per_window=80)

    fec_value = get_perturbed_free_energy_differences([fec])[0]

    u_ln = fec._form_u_ln()
    f_k = fec._compute_perturbed_free_energies(u_ln)
    print(f_k)


def test_max_nr_of_snapshots():
    from ..parameter_gradients import calculate_rmse_between_exp_and_calc
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x

    names = ['molDWRow_298', 'SAMPLmol2', 'SAMPLmol4']
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))

    env = 'vacuum'
    
    model = AlchemicalANI1ccx

    for nr_of_snapshots in [20, 80,120,150]:
        rmse = calculate_rmse_between_exp_and_calc(
            names,
            model = model,
            data_path=f"./data/{env}",
            env=env,
            bulk_energy_calculation=True,
            max_snapshots_per_window=nr_of_snapshots)
         
        



def test_calculate_rmse_between_exp_and_calc():
    from ..parameter_gradients import calculate_rmse_between_exp_and_calc, get_experimental_values
    from ..constants import kT
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x
    
    names = ['molDWRow_298', 'SAMPLmol2', 'SAMPLmol4']
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))

    env = 'vacuum'
    exp_values = get_experimental_values(names)
    
    rmse_list = []
    for model in [AlchemicalANI1ccx, AlchemicalANI2x, AlchemicalANI1x]:
        rmse = calculate_rmse_between_exp_and_calc(
            names,
            model = model,
            data_path=f"./data/{env}",
            env=env,
            bulk_energy_calculation=False,
            max_snapshots_per_window=100)
        assert(np.isclose((exp_results[names[2]]['energy'] * unit.kilocalorie_per_mole) / kT, exp_values[2].item()))
        rmse_list.append(rmse)


    print(exp_values.tolist())
    print(rmse_list)
    for e1, e2 in zip(exp_values.tolist()  , [1.8994317488369707, -10.232118388886946, -3.858011851547537]):
        assert(np.isclose(e1, e2))

    for e1, e2 in zip(rmse_list  , [5.662402629852295, 5.6707963943481445, 4.7712321281433105]):
        assert(np.isclose(e1, e2))


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_calculate_rmse_between_exp_and_calc_droplet():
    from ..parameter_gradients import calculate_rmse_between_exp_and_calc, get_experimental_values
    from ..constants import kT
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x

    rmse_list = []
    names = ['molDWRow_298']
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
    env = 'droplet'
    exp_values = get_experimental_values(names)
    diameter = 18
    assert (np.isclose(exp_values[0].item(), 1.8994317488369707))
    assert(np.isclose((exp_results[names[0]]['energy'] * unit.kilocalorie_per_mole) / kT, exp_values[0].item()))

    for model in [AlchemicalANI1ccx, AlchemicalANI2x, AlchemicalANI1x]:

        rmse = calculate_rmse_between_exp_and_calc(
            names=names,
            data_path=f"./data/{env}",
            model=model,
            env=env,
            bulk_energy_calculation=False,
            max_snapshots_per_window=10,
            diameter=diameter)
        rmse_list.append(rmse)

    print(rmse_list)
    for e1, e2 in zip(rmse_list  , [0.2889864338136634, 13.469154493717424, 7.909838568006453]):
        assert(np.isclose(e1, e2))


def test_calculate_mse():
    from ..parameter_gradients import calculate_mse
    import torch

    mse = calculate_mse(torch.tensor([1.0]), torch.tensor([4.0]))
    assert(mse == 9.)

    mse = calculate_mse(torch.tensor([1.0, 2.0]), torch.tensor([4.0,2.0]))
    assert(mse == 4.5)


def test_calculate_rmse():
    from ..parameter_gradients import calculate_rmse
    import numpy as np
    import torch

    rmse = calculate_rmse(torch.tensor([1.0]), torch.tensor([4.0]))
    assert(np.isclose(rmse, 3.))

    rmse = calculate_rmse(torch.tensor([1.0, 2.0]), torch.tensor([4.0,2.0]))
    assert(np.isclose(rmse, 2.1213))


def test_experimental_values():
    from ..parameter_gradients import get_experimental_values
    from ..constants import _get_names

    def compare_get_names():
        from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds
        with open('data/exp_results.pickle', 'rb') as f:
            exp_results = pickle.load(f)
        names = []
        for name in sorted(exp_results):
            if name in exclude_set_ANI + mols_with_charge + multiple_stereobonds:
                continue
            names.append(name)
        return names

    assert(_get_names() == compare_get_names())
    names = _get_names()
    assert(len(get_experimental_values(names)) == len(names))



@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_postprocessing_vacuum():
    from ..parameter_gradients import FreeEnergyCalculator, get_perturbed_free_energy_differences, get_experimental_values
    from ..constants import kT, device, exclude_set_ANI, mols_with_charge
    from ..parameter_gradients import setup_mbar
    from glob import glob
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x
    import numpy as np

    env = 'vacuum'
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
    names = ['molDWRow_298', 'SAMPLmol2', 'SAMPLmol4']


    for idx, model in enumerate([AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x ]):

        if idx == 0:
            fec_list = [setup_mbar(
                name,
                ANImodel=model,
                env=env,
                data_path='./data/vacuum',
                bulk_energy_calculation=True,
                max_snapshots_per_window=80) for name in names
                ]

            assert(len(fec_list) == 3)
            rmse = torch.sqrt(torch.mean((get_perturbed_free_energy_differences(fec_list) - get_experimental_values(names))**2))
            print([e.end_state_free_energy_difference[0] for e in fec_list])
            for fec, e2 in zip(fec_list, [-1.2104192392489894, -5.31605397264069, 4.055934972298076]):
                assert(np.isclose(fec.end_state_free_energy_difference[0], e2))        
            assert (np.isclose(rmse.item(), 5.393606768321977))
        
        
        elif idx == 1:
            fec_list = [setup_mbar(
                name,
                ANImodel=model,
                env=env,
                data_path='./data/vacuum',
                bulk_energy_calculation=True,
                max_snapshots_per_window=80) for name in names
                ]

            assert(len(fec_list) == 3)
            rmse = torch.sqrt(torch.mean((get_perturbed_free_energy_differences(fec_list) - get_experimental_values(names))**2))
            print([e.end_state_free_energy_difference[0] for e in fec_list])
            for fec, e2 in zip(fec_list, [-10.201508376053313, -9.919852168528479, 0.6758425107641388]):
                assert(np.isclose(fec.end_state_free_energy_difference[0], e2))        
            assert (np.isclose(rmse.item(), 5.464364003709803))

        elif idx == 2:           
            fec_list = [setup_mbar(
                name,
                ANImodel=model,
                env=env,
                data_path='./data/vacuum',
                bulk_energy_calculation=True,
                max_snapshots_per_window=50) for name in names
                ]
            assert(len(fec_list) == 3)
            rmse = torch.sqrt(torch.mean((get_perturbed_free_energy_differences(fec_list) - get_experimental_values(names))**2))
            print([e.end_state_free_energy_difference[0] for e in fec_list])
            for fec, e2 in zip(fec_list, [-7.6805827500672805, -9.655550628208003, 2.996804928927007]):
                assert(np.isclose(fec.end_state_free_energy_difference[0], e2))        
            assert (np.isclose(rmse.item(), 5.1878913627689895))


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_postprocessing_droplet():
    from ..parameter_gradients import FreeEnergyCalculator, get_perturbed_free_energy_differences, get_experimental_values
    from ..constants import kT, device, exclude_set_ANI, mols_with_charge
    from ..parameter_gradients import setup_mbar
    from glob import glob
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x

    for idx, model in enumerate([AlchemicalANI1ccx, AlchemicalANI1x]):

        env = 'droplet'
        exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
        names = ['molDWRow_298']
        diameter = 18
        
        fec_list = [
            setup_mbar(
                name,
                ANImodel=model,
                env=env,
                diameter=18,
                bulk_energy_calculation=False,
                data_path='./data/droplet',
                max_snapshots_per_window=10) for name in names
                ]

        if idx == 0:
            assert(len(fec_list) == 1)
            rmse = torch.sqrt(torch.mean((get_perturbed_free_energy_differences(fec_list) - get_experimental_values(names))**2))

            assert(np.isclose(fec_list[0].end_state_free_energy_difference[0].item(), -4.987183082930629))
            assert(np.isclose(rmse.item(),  5.992548148031282))
        
        elif idx == 1:
            assert(len(fec_list) == 1)
            rmse = torch.sqrt(torch.mean((get_perturbed_free_energy_differences(fec_list) - get_experimental_values(names))**2))
            assert(np.isclose(fec_list[0].end_state_free_energy_difference[0].item(), -13.765576744789922))
            assert(np.isclose(rmse.item(),  9.8578))

def _remove_files(name, max_epochs=1):
    os.remove(f'{name}.pt')
    for i in range(1, max_epochs):
        os.remove(f'{name}_{i}.pt')
    os.remove(f'{name}_best.pt')



@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_tweak_parameters_and_class_nn():
    # the tweaked parameters are stored as class variables
    # this can lead to some tricky situations.
    # It also means that whenever any optimization is performed, 
    # every new instance of the class has the new parameters
    from ..parameter_gradients import tweak_parameters
    import os
    from ..ani import AlchemicalANI1ccx

    names = ['molDWRow_298']
    max_epochs = 3

    # start with model 1
    model1 = AlchemicalANI1ccx([0,0])
    # save parameters at the beginning
    params_at_start_model1 = list(model1.tweaked_neural_network.parameters())[6][0].tolist()
    original_parameters_model1 = list(model1.original_neural_network.parameters())[6][0].tolist()

    # tweak parameters
    model, model_name = (AlchemicalANI1ccx, 'AlchemicalANI1ccx')
    rmse_training, rmse_val, rmse_test = tweak_parameters(
    env='vacuum',
    checkpoint_filename= f"{model_name}_vacuum.pt",
    names=names,
    ANImodel=model,
    max_snapshots_per_window=50,
    batch_size=1,
    data_path='./data/vacuum',
    nr_of_nn=8,
    load_checkpoint=False,
    max_epochs=max_epochs)

    _remove_files(model_name+'_vacuum', max_epochs)
    # get new tweaked parameters
    params_at_end_model1 = list(model1.tweaked_neural_network.parameters())[6][0].tolist()
    # make sure that somethign happend while tweaking
    assert (params_at_start_model1 != params_at_end_model1)
    # make sure that the starting parameters were the original paramters
    assert (params_at_start_model1 == original_parameters_model1)

    # initialize second model
    model2 = AlchemicalANI1ccx([0, 0])
    # get original parameters
    original_parameters_model2 = list(model2.original_neural_network.parameters())[6][0].tolist()
    # get tweaked parameters
    params_at_start_model2 = list(model2.tweaked_neural_network.parameters())[6][0].tolist()
    # tweaked parameters at start of model2 should be the same as at end of model1
    assert (params_at_start_model2 == params_at_end_model1)
    # tweaked parameters at start of model 1 are different than at start of model2
    assert (params_at_start_model1 != params_at_start_model2)
    # original parameters are the same
    assert (original_parameters_model1 == original_parameters_model2)




@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_tweak_parameters():
    from ..parameter_gradients import tweak_parameters_with_split
    import os
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x

    names = ['molDWRow_298', 'SAMPLmol2', 'SAMPLmol4']
    max_epochs=4
    for idx, (model, model_name) in enumerate(zip(
        [AlchemicalANI1ccx, AlchemicalANI2x, AlchemicalANI1x],
        ['AlchemicalANI1ccx', 'AlchemicalANI2x', 'AlchemicalANI1x'])):

        
        rmse_training, rmse_val, rmse_test = tweak_parameters_with_split(
        env='vacuum',
        checkpoint_filename= f"{model_name}_vacuum.pt",
        names=names,
        ANImodel=model,
        batch_size=3,
        data_path='./data/vacuum',
        max_snapshots_per_window=50,
        nr_of_nn=8,
        bulk_energy_calculation=True,
        max_epochs=max_epochs)

        if idx == 0:
            print(rmse_val)
            try:
                assert (np.isclose(rmse_val[-1], rmse_test))
                assert (np.isclose(rmse_val[0],  5.3938140869140625))
                assert (np.isclose(rmse_val[-1], 2.098975658416748))
            finally:
                _remove_files(model_name+'_vacuum', max_epochs)

        if idx == 1:
            print(rmse_val)
            try:

                assert(np.isclose(rmse_val[-1], rmse_test))
                assert (np.isclose(rmse_val[0],  5.187891006469727))
                assert (np.isclose(rmse_val[-1], 2.672308921813965))
            finally:
                _remove_files(model_name+'_vacuum', max_epochs)

        if idx == 2:
            print(rmse_val)
            try:
                assert(np.isclose(rmse_val[-1], rmse_test))
                assert (np.isclose(rmse_val[0],  4.582426071166992))
                assert (np.isclose(rmse_val[-1], 2.2336010932922363))
            finally:
                _remove_files(model_name+'_vacuum', max_epochs)



@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_tweak_parameters_droplet():
    from ..parameter_gradients import setup_and_perform_parameter_retraining
    import os
    from ..ani import AlchemicalANI1ccx, AlchemicalANI1x, AlchemicalANI2x

    names = ['molDWRow_298']
    env = 'droplet'
    diameter = 18
    max_epochs=2
    for idx, (model, model_name) in enumerate(zip(
        [AlchemicalANI1ccx, AlchemicalANI2x, AlchemicalANI1x],
        ['AlchemicalANI1ccx', 'AlchemicalANI2x', 'AlchemicalANI1x'])):
        rmse_training, rmse_val = setup_and_perform_parameter_retraining(
        env=env,
        names=names,
        ANImodel=model,
        batch_size=1,
        max_snapshots_per_window=5,
        checkpoint_filename= f"{model_name}_droplet.pt",
        data_path=f'./data/{env}',
        nr_of_nn=8,
        max_epochs=max_epochs,
        diameter=18)

        if idx == 0:
            try:
                assert(np.isclose(rmse_val[-1], rmse_test))
                assert (np.isclose(rmse_val[-1], 3.70445561408996))
            finally:
                _remove_files(model_name+'_droplet', max_epochs)
                print(rmse_training, rmse_val, rmse_test)

        elif idx == 1:
            try:
                assert(np.isclose(rmse_val[-1], rmse_test))
                assert(np.isclose(rmse_val[-1], 16.059447235919407))
            finally:
                _remove_files(model_name+'_droplet', max_epochs)
                print(rmse_training, rmse_val, rmse_test)

        elif idx == 2:
            try:
                assert(np.isclose(rmse_val[-1], rmse_test))
                assert(np.isclose(rmse_val[-1], 11.08650664572978))
            finally:
                _remove_files(model_name+'_droplet', max_epochs)
                print(rmse_training, rmse_val, rmse_test)
