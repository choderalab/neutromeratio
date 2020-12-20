"""
Unit and regression test for the neutromeratio package.
"""

# Import package, test suite, and other packages as needed
import neutromeratio
import pytest
import os
import pickle
import torch
from simtk import unit
import numpy as np
import mdtraj as md
from neutromeratio.constants import device
from openmmtools.utils import is_quantity_close
import pandas as pd
from rdkit import Chem
from neutromeratio.constants import device
from neutromeratio.utils import _get_traj


def test_restraint():
    from ..tautomers import Tautomer
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx

    name = "molDWRow_298"
    _, tautomer, _ = setup_alchemical_system_and_energy_function(
        name=name, env="vacuum", ANImodel=AlchemicalANI1ccx
    )

    atoms = tautomer.initial_state_ligand_atoms

    # testing BondHarmonicRestraint and BondFlatBottomRestraint for single coordinate set
    harmonic = neutromeratio.restraints.BondHarmonicRestraint(
        sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms
    )
    flat_bottom = neutromeratio.restraints.BondFlatBottomRestraint(
        sigma=0.1 * unit.angstrom, atom_i_idx=6, atom_j_idx=7, atoms=atoms
    )

    x0 = [
        [
            [-2.59013476, 0.33576663, 0.1207918],
            [-1.27753302, -0.24535815, -0.24640922],
            [-0.02964166, 0.2376853, 0.36130811],
            [1.14310714, -0.27818069, 0.0363466],
            [2.42889584, 0.19675597, 0.64147539],
            [1.16581937, -1.29903205, -0.90394092],
            [-1.26323767, -1.1673833, -1.09651496],
            [-3.34026286, -0.48284842, 0.08473526],
            [-2.57824303, 0.74964539, 1.15767283],
            [-2.91280517, 1.09171916, -0.61569433],
            [-0.01590019, 1.02814702, 1.09344649],
            [3.02354428, 0.80021351, -0.07878159],
            [2.26286222, 0.79203156, 1.54222352],
            [3.03941803, -0.72249508, 0.84622154],
            [0.94411149, -1.03666685, -1.85652944],
        ]
    ] * unit.angstrom

    coordinates = torch.tensor(
        x0.value_in_unit(unit.angstrom),
        requires_grad=True,
        device=device,
        dtype=torch.float32,
    )

    e1 = harmonic.restraint(
        coordinates
    ).item()  # restraint() takes coordinates in Angstrom
    e2 = flat_bottom.restraint(
        coordinates
    ).item()  # restraint() takes coordinates in Angstrom
    print("Restraing: {}.".format(e1))
    print("Restraing: {}.".format(e2))
    assert np.isclose(e1, 267.8730557591876574, rtol=1e-5)
    assert np.isclose(e2, 141.6041861927414, rtol=1e-5)

    # testing BondHarmonicRestraint and BondFlatBottomRestraint for batch coordinate set

    x0 = [
        [
            [-2.59013476, 0.33576663, 0.1207918],
            [-1.27753302, -0.24535815, -0.24640922],
            [-0.02964166, 0.2376853, 0.36130811],
            [1.14310714, -0.27818069, 0.0363466],
            [2.42889584, 0.19675597, 0.64147539],
            [1.16581937, -1.29903205, -0.90394092],
            [-1.26323767, -1.1673833, -1.09651496],
            [-3.34026286, -0.48284842, 0.08473526],
            [-2.57824303, 0.74964539, 1.15767283],
            [-2.91280517, 1.09171916, -0.61569433],
            [-0.01590019, 1.02814702, 1.09344649],
            [3.02354428, 0.80021351, -0.07878159],
            [2.26286222, 0.79203156, 1.54222352],
            [3.03941803, -0.72249508, 0.84622154],
            [0.94411149, -1.03666685, -1.85652944],
        ],
        [
            [-2.59013476, 0.3, 0.1207918],
            [-1.27753302, -0.24, -0.24640922],
            [-0.02964166, 0.23, 0.36130811],
            [1.14310714, -0.2, 0.0363466],
            [2.42889584, 0.2, 0.64147539],
            [1.16581937, -1.3, -0.90394092],
            [-1.26323767, -1.2, -1.09651496],
            [-3.34026286, -0.48, 0.08473526],
            [-2.57824303, 0.74, 1.15767283],
            [-2.91280517, 1.09, -0.61569433],
            [-0.01590019, 1.02, 1.09344649],
            [3.02354428, 0.80, -0.07878159],
            [2.26286222, 0.79, 1.54222352],
            [3.03941803, -0.72, 0.84622154],
            [0.94411149, -1.03, -1.85652944],
        ],
    ] * unit.angstrom

    coordinates = torch.tensor(
        x0.value_in_unit(unit.angstrom),
        requires_grad=True,
        device=device,
        dtype=torch.float32,
    )

    e1 = harmonic.restraint(coordinates).detach()
    e2 = flat_bottom.restraint(coordinates).detach()
    print("Restraing: {}.".format(e1))  # restraint() takes coordinates in Angstrom
    print("Restraing: {}.".format(e2))  # restraint() takes coordinates in Angstrom
    assert np.isclose(e1[0], 267.8731, rtol=1e-5)
    assert np.isclose(e2[0], 141.6042, rtol=1e-5)

    assert np.isclose(e1[1], 271.5412, rtol=1e-5)
    assert np.isclose(e2[1], 144.2746, rtol=1e-5)

    # test COM restraint for batch

    coordinates = torch.tensor(
        x0.value_in_unit(unit.nanometer),
        requires_grad=True,
        device=device,
        dtype=torch.float32,
    )

    center = np.array([0.0, 0.0, 0.0]) * unit.angstrom

    com_restraint = neutromeratio.restraints.CenterOfMassFlatBottomRestraint(
        sigma=0.1 * unit.angstrom,
        atom_idx=tautomer.hybrid_ligand_idxs[:-2],
        atoms=tautomer.hybrid_atoms[:-2],
        point=center,
    )

    e1 = com_restraint.restraint(
        coordinates
    ).detach()  # restraint() takes coordinates in Angstrom
    print("Restraing: {}.".format(e1))
    assert np.isclose(e1[0], 0.0, rtol=1e-3)
    assert np.isclose(e1[1], 0.0, rtol=1e-3)


def test_restraint_with_AlchemicalANI1ccx():
    import numpy as np
    from ..constants import kT
    from ..ani import AlchemicalANI1ccx
    from ..analysis import setup_alchemical_system_and_energy_function

    # generate tautomer object
    name = "molDWRow_298"
    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name, env="vacuum", ANImodel=AlchemicalANI1ccx
    )

    # read in pregenerated traj
    traj_path = (
        "data/test_data/vacuum/molDWRow_298/molDWRow_298_lambda_0.0000_in_vacuum.dcd"
    )
    top_path = "data/test_data/vacuum/molDWRow_298/molDWRow_298.pdb"
    traj, _ = _get_traj(traj_path, top_path)
    x0 = [x.xyz[0] for x in traj[0]] * unit.nanometer

    energy = energy_function.calculate_energy(x0, lambda_value=0.0)
    e_ = energy.energy.value_in_unit(unit.kilojoule_per_mole)
    print(e_)
    print(energy)
    assert is_quantity_close(
        energy.energy.in_units_of(unit.kilojoule_per_mole),
        (-906912.01647632 * unit.kilojoule_per_mole),
        rtol=1e-9,
    )


def test_restraint_with_AlchemicalANI1ccx_for_batches_in_vacuum():
    import numpy as np
    from ..constants import kT
    from ..ani import AlchemicalANI1ccx
    from ..analysis import setup_alchemical_system_and_energy_function

    # generate smiles
    name = "molDWRow_298"
    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name, env="vacuum", ANImodel=AlchemicalANI1ccx
    )

    # read in pregenerated traj
    traj_path = (
        "data/test_data/vacuum/molDWRow_298/molDWRow_298_lambda_0.0000_in_vacuum.dcd"
    )
    top_path = "data/test_data/vacuum/molDWRow_298/molDWRow_298.pdb"
    traj, _ = _get_traj(traj_path, top_path)
    x0 = [x.xyz[0] for x in traj[:10]] * unit.nanometer

    energy = energy_function.calculate_energy(x0, lambda_value=0.0)
    e_ = energy.energy.value_in_unit(unit.kilojoule_per_mole)
    comp_ = [
        -906912.01647632,
        -906874.76383595,
        -906873.86032954,
        -906880.6874288,
        -906873.53381254,
        -906867.20345442,
        -906862.08888594,
        -906865.00135805,
        -906867.07548311,
        -906871.27971175,
    ]
    for e, comp_e in zip(e_, comp_):
        np.isclose(e, comp_e, rtol=1e-6)


def test_COM_restraint_with_AlchemicalANI1ccx_for_batches_in_droplet():
    import numpy as np
    from ..constants import kT
    from ..ani import AlchemicalANI2x
    from ..analysis import setup_alchemical_system_and_energy_function

    # generate smiles
    name = "molDWRow_298"
    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name,
        env="droplet",
        ANImodel=AlchemicalANI2x,
        diameter=10,
        base_path=f"data/test_data/droplet/{name}",
    )

    # read in pregenerated traj
    traj_path = (
        "data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
    )
    top_path = "data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    traj, _ = _get_traj(traj_path, top_path)
    x0 = [x.xyz[0] for x in traj[:10]] * unit.nanometer

    coordinates = torch.tensor(
        x0.value_in_unit(unit.nanometers),
        requires_grad=True,
        device=device,
        dtype=torch.float32,
    )

    r = energy_function._compute_restraint_bias(coordinates, lambda_value=0.0)
    r = r.detach().tolist()

    full_restraints = [
        0.4100764785669023,
        0.03667892693933966,
        0.0464554783605855,
        1.6675558008972122,
        1.6304614437658922,
        2.7633326202414183,
        3.7763598532698115,
        9.16871744958779,
        5.977379445002653,
        3.429661035102869,
    ]
    np.isclose(r, full_restraints)

    print(energy_function.list_of_lambda_restraints[-1])

    coordinates = torch.tensor(
        x0.value_in_unit(unit.angstrom),
        requires_grad=True,
        device=device,
        dtype=torch.float32,
    )

    com_restraint = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    com_r = (
        energy_function.list_of_lambda_restraints[-1].restraint(coordinates).tolist()
    )
    np.isclose(com_restraint, com_r)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_droplet_bottom_restraint_with_AlchemicalANI1ccx_for_batches():
    import numpy as np
    from ..constants import kJ_mol_to_kT
    from ..ani import AlchemicalANI2x
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..restraints import CenterFlatBottomRestraint

    name = "molDWRow_298"
    energy_function, _, _ = setup_alchemical_system_and_energy_function(
        name=name,
        env="droplet",
        ANImodel=AlchemicalANI2x,
        diameter=10,
        base_path=f"data/test_data/droplet/{name}",
    )

    # read in pregenerated traj
    traj_path = (
        "data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
    )
    top_path = "data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    traj, _ = _get_traj(traj_path, top_path)
    x0 = [x.xyz[0] for x in traj[:10]] * unit.nanometer

    coordinates = torch.tensor(
        x0.value_in_unit(unit.nanometers),
        requires_grad=True,
        device=device,
        dtype=torch.float32,
    )

    r = energy_function._compute_restraint_bias(coordinates, lambda_value=0.0)
    r = r.detach().tolist()

    full_restraints = [
        0.4100764785669023,
        0.03667892693933966,
        0.0464554783605855,
        1.6675558008972122,
        1.6304614437658922,
        2.7633326202414183,
        3.7763598532698115,
        9.16871744958779,
        5.977379445002653,
        3.429661035102869,
    ]
    np.isclose(r, full_restraints)

    coordinates = torch.tensor(
        x0.value_in_unit(unit.angstrom),
        requires_grad=True,
        device=device,
        dtype=torch.float32,
    )
    nr_of_mols = len(coordinates)
    droplet_restraint = torch.tensor(
        [0.0] * nr_of_mols, device=device, dtype=torch.float64
    )

    for r in energy_function.list_of_lambda_restraints:
        if isinstance(r, CenterFlatBottomRestraint):
            c = r.restraint(coordinates)
            droplet_restraint += c * kJ_mol_to_kT
    print(droplet_restraint.tolist())

    only_droplet_restrain_energies = [
        0.19625303281218226,
        0.03667892693933966,
        0.0464554783605855,
        0.4731166597911941,
        1.6304614437658922,
        2.7633326202414183,
        3.7763598532698115,
        4.700956016051697,
        4.988977610418124,
        3.429661035102869,
    ]
    np.isclose(droplet_restraint.tolist(), only_droplet_restrain_energies, rtol=1e-6)


def test_restraint_with_AlchemicalANI1ccx_for_batches_in_droplet():
    import numpy as np
    from ..constants import kT
    from ..ani import AlchemicalANI2x
    from ..analysis import setup_alchemical_system_and_energy_function

    # generate smiles
    name = "molDWRow_298"
    energy_function, _, _ = setup_alchemical_system_and_energy_function(
        name=name,
        env="droplet",
        ANImodel=AlchemicalANI2x,
        diameter=10,
        base_path=f"data/test_data/droplet/{name}",
    )

    # read in pregenerated traj
    traj_path = (
        "data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
    )
    top_path = "data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    traj, _ = _get_traj(traj_path, top_path)
    x0 = [x.xyz[0] for x in traj[:10]] * unit.nanometer

    energy = energy_function.calculate_energy(x0, lambda_value=0.0)
    e_ = list(energy.energy.value_in_unit(unit.kilojoule_per_mole))
    comp_ = [
        -3515628.7141218423,
        -3515543.79281359,
        -3515512.8337717773,
        -3515512.0131278695,
        -3515511.2858314235,
        -3515511.58966059,
        -3515504.9864817774,
        -3515505.5672234907,
        -3515498.497307367,
        -3515488.45506095,
    ]
    for e, comp_e in zip(e_, comp_):
        np.isclose(float(e), float(comp_e), rtol=1e-9)
