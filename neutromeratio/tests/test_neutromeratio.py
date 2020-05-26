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

    print(os.getcwd())
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

    # test if droplet works for
    t.add_droplet(t.final_state_ligand_topology, t.final_state_ligand_coords[0])


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

    # test if droplet works for
    t.add_droplet(t.final_state_ligand_topology, t.final_state_ligand_coords[0])

def test_neutromeratio_energy_calculations_with_torchANI_model():

    from neutromeratio.tautomers import Tautomer
    from neutromeratio.constants import kT
    import numpy as np

    # read in exp_results.pickle
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    # generate smiles
    name = 'molDWRow_298'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles)
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    # read in pregenerated traj
    traj_path = 'neutromeratio/data/molDWRow_298_lambda_0.0000_kappa_0.0000.dcd'
    top_path = 'neutromeratio/data/molDWRow_298_lambda_0.0000_kappa_0.0000.pdb'
    traj = md.load(traj_path, top=md.load(top_path).topology)
    traj = traj.atom_slice([a for a in range(len(tautomer.initial_state_ligand_atoms))])
    # TODO: update this a bit
    # overwrite the coordinates that rdkit generated with the first frame in the traj
    model = neutromeratio.ani.PureANI1ccx()
    torch.set_num_threads(1)
    energy_function = neutromeratio.ANI1_force_and_energy(
        model=model,
        atoms=tautomer.initial_state_ligand_atoms,
        mol=None)
    coordinates = [x.xyz[0] for x in traj[:10]] * unit.nanometer
    energy = energy_function.calculate_energy(coordinates)

    # overwrite the coordinates that rdkit generated with the first frame in the traj
    x0 = [x.xyz[0] for x in traj[0]] * unit.nanometer
    model = neutromeratio.ani.PureANI1ccx()
    torch.set_num_threads(1)
    energy_function = neutromeratio.ANI1_force_and_energy(
        model=model,
        atoms=tautomer.initial_state_ligand_atoms,
        mol=None)

    energy = energy_function.calculate_energy(x0)
    energy.energy.value_in_unit(unit.kilojoule_per_mole)
    assert(is_quantity_close(energy.energy, (-906911.9843514563 * unit.kilojoule_per_mole), rtol=1e-5))

    tautomer = tautomers[1]
    tautomer.perform_tautomer_transformation()

    # read in pregenerated traj
    traj_path = 'neutromeratio/data/molDWRow_298_lambda_0.0000_kappa_1.0000.dcd'
    top_path = 'neutromeratio/data/molDWRow_298_lambda_0.0000_kappa_1.0000.pdb'
    traj = md.load(traj_path, top=md.load(top_path).topology)
    traj = traj.atom_slice([a for a in range(len(tautomer.initial_state_ligand_atoms))])

    # overwrite the coordinates that rdkit generated with the first frame in the traj
    x0 = [x.xyz[0] for x in traj[0]] * unit.nanometer
    model = neutromeratio.ani.PureANI1ccx()
    torch.set_num_threads(1)
    energy_function = neutromeratio.ANI1_force_and_energy(
        model=model,
        atoms=tautomer.initial_state_ligand_atoms,
        mol=None)

    energy = energy_function.calculate_energy(x0)
    assert(is_quantity_close(energy.energy, (-906920.2981953777 * unit.kilojoule_per_mole), rtol=1e-5))


def test_neutromeratio_energy_calculations_LinearAlchemicalSingleTopologyANI_model():
    from ..tautomers import Tautomer
    import numpy as np
    from ..constants import kT

    # read in exp_results.pickle
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    # generate smiles
    name = 'molDWRow_298'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles)
    ######################################################################
    ######################################################################
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    # read in pregenerated traj
    traj_path = 'neutromeratio/data/molDWRow_298_lambda_0.0000_kappa_0.0000.dcd'
    top_path = 'neutromeratio/data/molDWRow_298_lambda_0.0000_kappa_0.0000.pdb'
    traj = md.load(traj_path, top=md.load(top_path).topology)

    # generate tautomer transformation
    dummy_atoms = [tautomer.hybrid_hydrogen_idx_at_lambda_1, tautomer.hybrid_hydrogen_idx_at_lambda_0]
    atoms = tautomer.hybrid_atoms
    # overwrite the coordinates that rdkit generated with the first frame in the traj
    model = neutromeratio.ani.LinearAlchemicalSingleTopologyANI(alchemical_atoms=dummy_atoms)
    torch.set_num_threads(1)

    energy_function = neutromeratio.ANI1_force_and_energy(
        model=model,
        atoms=atoms,
        mol=None)

    x0 = [x.xyz[0] for x in traj[0]] * unit.nanometer

    energy = energy_function.calculate_energy(x0, lambda_value=1.0)
    assert(is_quantity_close(energy.energy, (-906630.9281008451 * unit.kilojoule_per_mole), rtol=1e-9))
    energy = energy_function.calculate_energy(x0, lambda_value=0.0)
    assert(is_quantity_close(energy.energy, (-906911.9843514563 * unit.kilojoule_per_mole), rtol=1e-9))
    ######################################################################
    ######################################################################
    tautomer = tautomers[1]
    tautomer.perform_tautomer_transformation()

    # read in pregenerated traj
    traj_path = 'neutromeratio/data/molDWRow_298_lambda_0.0000_kappa_1.0000.dcd'
    top_path = 'neutromeratio/data/molDWRow_298_lambda_0.0000_kappa_1.0000.pdb'
    traj = md.load(traj_path, top=md.load(top_path).topology)

    # generate tautomer transformation
    dummy_atoms = [tautomer.hybrid_hydrogen_idx_at_lambda_1, tautomer.hybrid_hydrogen_idx_at_lambda_0]
    atoms = tautomer.hybrid_atoms
    # overwrite the coordinates that rdkit generated with the first frame in the traj
    model = neutromeratio.ani.LinearAlchemicalSingleTopologyANI(alchemical_atoms=dummy_atoms)
    torch.set_num_threads(1)

    energy_function = neutromeratio.ANI1_force_and_energy(
        model=model,
        atoms=atoms,
        mol=None)

    x0 = [x.xyz[0] for x in traj[0]] * unit.nanometer

    energy = energy_function.calculate_energy(x0, lambda_value=1.0)
    assert(is_quantity_close(energy.energy, (-906700.3482745718 * unit.kilojoule_per_mole), rtol=1e-9))
    energy = energy_function.calculate_energy(x0, lambda_value=0.0)
    assert(is_quantity_close(energy.energy, (-906920.2981953777 * unit.kilojoule_per_mole), rtol=1e-9))

    # test the batch coordinates -- traj is a list of unit'd coordinates
    coordinates = [x.xyz[0] for x in traj[:10]] * unit.nanometer
    energy=energy_function.calculate_energy(coordinates)
    for e1, e2 in zip(energy.energy, [-906920.298,-906905.387,-906894.271,-906894.193,-906897.663,-906893.282,-906892.392,-906891.93,-906892.034,-906894.464] * unit.kilojoule_per_mole):
        assert(is_quantity_close(e1, e2, rtol=1e-3))    
    for e1, e2 in zip(energy.restraint_bias, [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] * unit.kilojoule_per_mole):
        assert(is_quantity_close(e1, e2, rtol=1e-3))    
    for e1, e2 in zip(energy.stddev, [3.74471131, 4.1930047 , 3.38845079, 3.93200761, 3.19887848,
       4.02611676, 4.32329868, 2.92180683, 4.3240609 , 2.78107752]* unit.kilojoule_per_mole):
        assert(is_quantity_close(e1, e2, rtol=1e-3))    
    
def test_restraint():
    from neutromeratio.tautomers import Tautomer
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    name = 'molDWRow_298'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles)
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    # read in pregenerated traj
    traj_path = 'neutromeratio/data/molDWRow_298_lambda_0.0000_kappa_1.0000.dcd'
    top_path = 'neutromeratio/data/molDWRow_298_lambda_0.0000_kappa_1.0000.pdb'
    traj = md.load(traj_path, top=md.load(top_path).topology)
    traj = traj.atom_slice([a for a in range(len(tautomer.initial_state_ligand_atoms))])

    atoms = tautomer.initial_state_ligand_atoms
    harmonic = neutromeratio.restraints.BondHarmonicRestraint(
        sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms)
    flat_bottom = neutromeratio.restraints.BondFlatBottomRestraint(
        sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms)

    coordinates = torch.tensor([tautomer.initial_state_ligand_coords[0].value_in_unit(unit.nanometer)],
                               requires_grad=True, device=device, dtype=torch.float32)

    print('Restraing: {}.'.format(harmonic.restraint(coordinates)))
    print('Restraing: {}.'.format(flat_bottom.restraint(coordinates)))


def test_restraint_with_LinearAlchemicalSingleTopologyANI():
    import numpy as np
    from neutromeratio.constants import kT
    # read in exp_results.pickle
    with open('data/exp_results.pickle', 'rb') as f:
        exp_results = pickle.load(f)

    # generate smiles
    name = 'molDWRow_298'
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles)
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    # read in pregenerated traj
    traj_path = 'neutromeratio/data/molDWRow_298_lambda_0.0000_kappa_0.0000.dcd'
    top_path = 'neutromeratio/data/molDWRow_298_lambda_0.0000_kappa_0.0000.pdb'
    traj = md.load(traj_path, top=md.load(top_path).topology)
    x0 = [x.xyz[0] for x in traj[0]] * unit.nanometer

    # the first of the alchemical_atoms will be dummy at lambda 0, the second at lambda 1
    # protocoll goes from 0 to 1
    dummy_atoms = [tautomer.hybrid_hydrogen_idx_at_lambda_1, tautomer.hybrid_hydrogen_idx_at_lambda_0]
    atoms = tautomer.hybrid_atoms

    model = neutromeratio.ani.LinearAlchemicalSingleTopologyANI(alchemical_atoms=dummy_atoms)

    energy_function = neutromeratio.ANI1_force_and_energy(
        model=model,
        atoms=atoms,
        mol=None)

    energy_function.list_of_restraints = tautomer.ligand_restraints

    energy = energy_function.calculate_energy(x0,lambda_value=0.0)
    assert (is_quantity_close(energy.energy, (-906911.9843514563 * unit.kilojoule_per_mole), rtol=1e-9))


def test_min_and_single_point_energy():

    # name of the system
    name = 'molDWRow_298'

    # extract smiles
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles)
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    # set model
    model = neutromeratio.ani.PureANI1ccx()
    model = model.to(device)
    torch.set_num_threads(1)

    # calculate energy using both structures and pure ANI1ccx
    for ase_mol, ligand_atoms, ligand_coords in zip([tautomer.initial_state_ase_mol, tautomer.final_state_ase_mol],
                                                    [tautomer.initial_state_ligand_atoms, tautomer.final_state_ligand_atoms],
                                                    [tautomer.initial_state_ligand_coords, tautomer.final_state_ligand_coords]):
        energy_function = neutromeratio.ANI1_force_and_energy(
            model=model,
            atoms=ligand_atoms,
            mol=ase_mol)

        for coords in ligand_coords:
            # minimize
            # put coordinates back in [K][3] *unit format
            x0, hist_e = energy_function.minimize(coords)
            print(energy_function.calculate_energy([x0/unit.angstrom] * unit.angstrom).energy)


def test_thermochemistry():

    # name of the system
    name = 'molDWRow_298'

    # extract smiles
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles)
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    # set model
    model = neutromeratio.ani.PureANI1ccx()
    model = model.to(device)
    torch.set_num_threads(1)

    # calculate energy using both structures and pure ANI1ccx
    for ase_mol, ligand_atoms, ligand_coords in zip([tautomer.initial_state_ase_mol, tautomer.final_state_ase_mol],
                                                    [tautomer.initial_state_ligand_atoms, tautomer.final_state_ligand_atoms],
                                                    [tautomer.initial_state_ligand_coords, tautomer.final_state_ligand_coords]):
        energy_function = neutromeratio.ANI1_force_and_energy(
            model=model,
            atoms=ligand_atoms,
            mol=ase_mol,
        )
        for coords in ligand_coords:
            # minimize
            # put coordinates back in [1][K][3] *unit format
            x0, hist_e = energy_function.minimize(coords)
            energy_function.get_thermo_correction(x0) # x has [1][K][3] dimenstion -- N: number of mols, K: number of atoms


def test_euqilibrium():
    # name of the system
    name = 'molDWRow_298'
    # number of steps
    n_steps = 50

    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))

    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles)
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    # define the alchemical atoms
    alchemical_atoms = [tautomer.hybrid_hydrogen_idx_at_lambda_1,
                        tautomer.hybrid_hydrogen_idx_at_lambda_0]

    # extract hydrogen donor idx and hydrogen idx for from_mol
    model = neutromeratio.ani.LinearAlchemicalSingleTopologyANI(alchemical_atoms=alchemical_atoms)
    model = model.to(device)
    torch.set_num_threads(1)

    # perform initial sampling
    energy_function = neutromeratio.ANI1_force_and_energy(
        model=model,
        atoms=tautomer.hybrid_atoms,
        mol=None
    )

    for e in tautomer.ligand_restraints + tautomer.hybrid_ligand_restraints:
        print(e)
        energy_function.add_restraint_to_lambda_protocol(e)

    x0 = [np.array(tautomer.hybrid_coords)] * unit.angstrom # format [1][K][3] * unit
    x0, hist_e = energy_function.minimize(x0)

    energy_and_force = lambda x : energy_function.calculate_force(x, 1.0)

    langevin = neutromeratio.LangevinDynamics(atoms=tautomer.hybrid_atoms,
                                              energy_and_force=energy_and_force,
                                              )

    equilibrium_samples, energies, restraint_bias, stddev, ensemble_bias = langevin.run_dynamics(x0,
                                                                                                 n_steps=n_steps,
                                                                                                 stepsize=1.0 * unit.femtosecond,
                                                                                                 progress_bar=True)

    energy_and_force = lambda x : energy_function.calculate_force(x, 0.0)

    langevin = neutromeratio.LangevinDynamics(atoms=tautomer.hybrid_atoms,
                                              energy_and_force=energy_and_force)

    equilibrium_samples, energies, restraint_bias, stddev, ensemble_bias = langevin.run_dynamics(x0,
                                                                                                 n_steps=n_steps,
                                                                                                 stepsize=1.0 * unit.femtosecond,
                                                                                                 progress_bar=True)


def test_tautomer_conformation():
    # name of the system
    name = 'molDWRow_298'
    # number of steps

    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))

    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles, nr_of_conformations=5)
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    print(f"Nr of initial conformations: {tautomer.initial_state_ligand_coords}")
    print(f"Nr of final conformations: {tautomer.final_state_ligand_coords}")

    assert(len(tautomer.initial_state_ligand_coords) == 5)
    assert(len(tautomer.final_state_ligand_coords) == 5)

    # set model
    model = neutromeratio.ani.PureANI1ccx()
    model = model.to(device)
    torch.set_num_threads(1)

    # calculate energy using both structures and pure ANI1ccx
    for ase_mol, ligand_atoms, ligand_coords in zip([tautomer.initial_state_ase_mol, tautomer.final_state_ase_mol], [tautomer.initial_state_ligand_atoms, tautomer.final_state_ligand_atoms], [tautomer.initial_state_ligand_coords, tautomer.final_state_ligand_coords]):
        energy_function = neutromeratio.ANI1_force_and_energy(
            model=model,
            atoms=ligand_atoms,
            mol=ase_mol,
        )

        for n_conf, coords in enumerate(ligand_coords):
            # minimize
            print(f"Conf: {n_conf}")
            x, e_min_history = energy_function.minimize(coords, maxiter=100000)
            energy = energy_function.calculate_energy([x/unit.angstrom] * unit.angstrom) # coordinates need to be in [N][K][3] format
            e_correction = energy_function.get_thermo_correction(x)
            print(f"Energy: {energy.energy}")
            print(f"Energy ensemble stddev: {energy.stddev}")
            print(f"Energy correction: {e_correction}")


def test_mining_minima():
    # name of the system
    name = 'molDWRow_298'
    # number of steps
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
    torch.set_num_threads(1)

    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles)
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
    from neutromeratio.constants import kT
    import numpy as np

    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))

    name = 'molDWRow_298'

    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']

    # generate both rdkit mol
    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles, nr_of_conformations=5)
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()
    m = tautomer.add_droplet(tautomer.hybrid_topology,
                             tautomer.hybrid_coords,
                             diameter=16 * unit.angstrom,
                             restrain_hydrogen_bonds=True,
                             top_file=f"data/{name}_in_droplet.pdb")

    # define the alchemical atoms
    alchemical_atoms = [tautomer.hybrid_hydrogen_idx_at_lambda_1, tautomer.hybrid_hydrogen_idx_at_lambda_0]

    # extract hydrogen donor idx and hydrogen idx for from_mol
    model = neutromeratio.ani.LinearAlchemicalSingleTopologyANI(alchemical_atoms=alchemical_atoms)
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

    energy = energy_function.calculate_energy([tautomer.ligand_in_water_coordinates/unit.angstrom]*unit.angstrom)
    assert(is_quantity_close(energy.energy[0], (-15547479.771537919 * unit.kilojoule_per_mole), rtol=1e-7))


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Psi4 import fails on travis."
)
def test_psi4():
    from neutromeratio import qm
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


def test_parameter_gradient():
    from ..constants import mols_with_charge, exclude_set_ANI, kT
    from tqdm import tqdm
    from ..parameter_gradients import FreeEnergyCalculator
    from ..analysis import setup_energy_function
    
    # TODO: pkg_resources instead of filepath relative to execution directory
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
    # nr of steps
    #################
    n_steps = 100
    #################

    # specify the system you want to simulate
    name = 'molDWRow_298'  #Experimental free energy difference: 1.132369 kcal/mol
    if name in exclude_set_ANI + mols_with_charge:
        raise RuntimeError(f"{name} is part of the list of excluded molecules. Aborting")

    energy_function, tautomer, flipped = setup_energy_function(name)
    
    x0 = tautomer.hybrid_coords
    potential_energy_trajs = []
    ani_trajs = []
    lambdas = np.linspace(0, 1, 5)

    for lamb in tqdm(lambdas):
        # minimize coordinates with a given lambda value
        x0, e_history = energy_function.minimize(x0, maxiter=5000, lambda_value=lamb)
        # define energy function with a given lambda value
        energy_and_force = lambda x : energy_function.calculate_force(x, lamb)
        # define langevin object with a given energy function
        langevin = neutromeratio.LangevinDynamics(atoms=tautomer.hybrid_atoms,
                                        energy_and_force=energy_and_force)

        # sampling
        equilibrium_samples, energies, restraint_bias, stddev, ensemble_bias = langevin.run_dynamics(x0,
                                                                        n_steps=n_steps,
                                                                        stepsize=1.0*unit.femtosecond,
                                                                        progress_bar=False)
        potential_energy_trajs.append(energies)

        ani_trajs.append(md.Trajectory([x / unit.nanometer for x in equilibrium_samples], tautomer.hybrid_topology))

    # calculate free energy in kT
    fec = FreeEnergyCalculator(ani_model=energy_function,
                               md_trajs=ani_trajs,
                               potential_energy_trajs=potential_energy_trajs,
                               lambdas=lambdas,
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
    params = list(energy_function.model.neural_networks.parameters())
    none_counter = 0
    for p in params:
        if(p.grad == None):  # some are None!
            none_counter += 1

    assert(len(params) == 256)
    assert (none_counter == 64)
    
def test_postprocessing():
    from ..parameter_gradients import FreeEnergyCalculator, get_free_energy_differences, get_experimental_values
    from ..constants import kT, device, exclude_set_ANI, mols_with_charge
    from glob import glob

    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
    names = ['molDWRow_298', 'SAMPLmol2', 'SAMPLmol4']
    setup_mbar = neutromeratio.analysis.setup_mbar
    fec_list = [setup_mbar(name, './data/', thinning = 50, max_snapshots_per_window = -1) for name in names]

    assert(len(fec_list) == 3)
    rmse = torch.sqrt(torch.mean((get_free_energy_differences(fec_list) - get_experimental_values(names))**2))

    assert(np.isclose(fec_list[0].end_state_free_energy_difference[0], -1.2657010719456991, rtol=1.e-2))
    assert(np.isclose(fec_list[1].end_state_free_energy_difference[0], -4.764917445894416, rtol=1.e-2))
    assert(np.isclose(fec_list[2].end_state_free_energy_difference[0], 4.127431117241131, rtol=1.e-2))
    assert(np.isclose(rmse.item(),  5.599380922019047, rtol=1.e-2))

def test_tweak_parameters():
    from ..parameter_gradients import tweak_parameters
    import os
    names = ['molDWRow_298', 'SAMPLmol2', 'SAMPLmol4']

    rmse_training, rmse_val, rmse_test = tweak_parameters(names=names, batch_size=3, data_path='./data', nr_of_nn=8, max_epochs=2)
    try:
        os.remove('best.pt')
        os.remove('latest.pt')

    except FileNotFoundError:
        pass
    
    np.isclose(rmse_val[-1], rmse_test, rtol=1e-2)
    np.isclose(rmse_val[-1], 5.279, rtol=1e-2)
