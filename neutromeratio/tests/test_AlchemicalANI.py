"""
Unit and regression test for the neutromeratio package.
"""

# Import package, test suite, and other packages as needed
import neutromeratio
import pytest
import os
import pickle
from simtk import unit
from neutromeratio.constants import device
import torchani
from openmmtools.utils import is_quantity_close
from neutromeratio.constants import device
from neutromeratio.utils import _get_traj


def test_neutromeratio_energy_calculations_with_AlchemicalANI2x_in_vacuum():

    from ..tautomers import Tautomer
    from ..constants import kT
    from ..analysis import setup_alchemical_system_and_energy_function
    import numpy as np
    from ..ani import AlchemicalANI2x, ANI1ccx, ANI2x, AlchemicalANI1ccx

    # vacuum system
    name = "molDWRow_298"
    energy_function, tautomer, _ = setup_alchemical_system_and_energy_function(
        name=name, ANImodel=AlchemicalANI2x, env="vacuum"
    )

    # read in pregenerated traj
    traj_path = (
        "data/test_data/vacuum/molDWRow_298/molDWRow_298_lambda_0.0000_in_vacuum.dcd"
    )
    top_path = "data/test_data/vacuum/molDWRow_298/molDWRow_298.pdb"

    # test energies with neutromeratio ANI objects
    # with ANI2x
    model = neutromeratio.ani.ANI2x()
    # final state
    traj, _ = _get_traj(traj_path, top_path, [tautomer.hybrid_hydrogen_idx_at_lambda_1])
    coordinates = [x.xyz[0] for x in traj[:10]] * unit.nanometer
    assert len(tautomer.initial_state_ligand_atoms) == len(coordinates[0])
    energy_function = neutromeratio.ANI_force_and_energy(
        model=model, atoms=tautomer.final_state_ligand_atoms, mol=None
    )
    ANI2x_energy_final_state = energy_function.calculate_energy(
        coordinates,
    )
    # initial state
    traj, _ = _get_traj(traj_path, top_path, [tautomer.hybrid_hydrogen_idx_at_lambda_0])
    coordinates = [x.xyz[0] for x in traj[:10]] * unit.nanometer
    energy_function = neutromeratio.ANI_force_and_energy(
        model=model, atoms=tautomer.initial_state_ligand_atoms, mol=None
    )
    ANI2x_energy_initial_state = energy_function.calculate_energy(
        coordinates,
    )

    assert is_quantity_close(
        ANI2x_energy_initial_state.energy[0].in_units_of(unit.kilojoule_per_mole),
        (-907243.8987177598 * unit.kilojoule_per_mole),
        rtol=1e-5,
    )

    # compare with energies for AlchemicalANI object
    # vacuum system
    traj, _ = _get_traj(traj_path, top_path)
    coordinates = [x.xyz[0] for x in traj[:10]] * unit.nanometer

    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name, ANImodel=AlchemicalANI2x, env="vacuum"
    )
    AlchemicalANI2x_energy_lambda_0 = energy_function.calculate_energy(
        coordinates, lambda_value=0.0
    )
    AlchemicalANI2x_energy_lambda_1 = energy_function.calculate_energy(
        coordinates, lambda_value=1.0
    )

    # making sure that the AlchemicalANI and the ANI results are the same on the endstates (with removed dummy atoms)
    assert is_quantity_close(
        AlchemicalANI2x_energy_lambda_1.energy[0].in_units_of(unit.kilojoule_per_mole),
        (-907243.8987177598 * unit.kilojoule_per_mole),
        rtol=1e-5,
    )

    assert is_quantity_close(
        AlchemicalANI2x_energy_lambda_0.energy[0].in_units_of(unit.kilojoule_per_mole),
        ANI2x_energy_final_state.energy[0].in_units_of(unit.kilojoule_per_mole),
        rtol=1e-5,
    )


def test_break_up_AlchemicalANI2x_energies():

    from ..tautomers import Tautomer
    from ..constants import kT
    from ..analysis import setup_alchemical_system_and_energy_function
    import numpy as np
    from ..ani import AlchemicalANI2x, CompartimentedAlchemicalANI2x

    # vacuum system
    name = "molDWRow_298"

    # read in pregenerated traj
    traj_path = (
        "data/test_data/vacuum/molDWRow_298/molDWRow_298_lambda_0.0000_in_vacuum.dcd"
    )
    top_path = "data/test_data/vacuum/molDWRow_298/molDWRow_298.pdb"

    # test energies with neutromeratio AlchemicalANI objects
    # with ANI2x
    traj, _ = _get_traj(traj_path, top_path)
    coordinates = [x.xyz[0] for x in traj[:10]] * unit.nanometer

    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name, ANImodel=AlchemicalANI2x, env="vacuum"
    )

    AlchemicalANI2x_energy_lambda_0 = energy_function.calculate_energy(
        coordinates, lambda_value=0.0
    )

    AlchemicalANI2x_energy_lambda_1 = energy_function.calculate_energy(
        coordinates, lambda_value=1.0
    )

    # making sure that the AlchemicalANI and the ANI results are the same on the endstates (with removed dummy atoms)
    assert is_quantity_close(
        AlchemicalANI2x_energy_lambda_1.energy[0].in_units_of(unit.kilojoule_per_mole),
        (-907243.8987177598 * unit.kilojoule_per_mole),
        rtol=1e-5,
    )
    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name, ANImodel=CompartimentedAlchemicalANI2x, env="vacuum"
    )

    CompartimentedAlchemicalANI2x_1 = energy_function.calculate_energy(
        coordinates, lambda_value=1.0
    )

    # making sure that the AlchemicalANI and the CompartimentedAlchemicalANI2x results are the same
    assert is_quantity_close(
        CompartimentedAlchemicalANI2x_1.energy[0].in_units_of(unit.kilojoule_per_mole),
        (-907243.8987177598 * unit.kilojoule_per_mole),
        rtol=1e-5,
    )

    CompartimentedAlchemicalANI2x_0 = energy_function.calculate_energy(
        coordinates, lambda_value=0.0
    )
    print(CompartimentedAlchemicalANI2x_0.energy[0])

    # making sure that the AlchemicalANI and the CompartimentedAlchemicalANI2x results are the same
    assert is_quantity_close(
        CompartimentedAlchemicalANI2x_0.energy[0].in_units_of(unit.kilojoule_per_mole),
        AlchemicalANI2x_energy_lambda_0.energy[0].in_units_of(unit.kilojoule_per_mole),
        rtol=1e-5,
    )


def test_break_up_AlchemicalANI2x_timings():

    from ..tautomers import Tautomer
    from ..constants import kT
    from ..analysis import setup_alchemical_system_and_energy_function
    import numpy as np
    from ..ani import AlchemicalANI2x, CompartimentedAlchemicalANI2x

    # vacuum system
    name = "molDWRow_298"

    # read in pregenerated traj
    traj_path = (
        "data/test_data/vacuum/molDWRow_298/molDWRow_298_lambda_0.0000_in_vacuum.dcd"
    )
    top_path = "data/test_data/vacuum/molDWRow_298/molDWRow_298.pdb"

    # test energies with neutromeratio AlchemicalANI objects
    # with ANI2x
    traj, _ = _get_traj(traj_path, top_path)
    coordinates = [x.xyz[0] for x in traj[:100]] * unit.nanometer

    energy_function, _, _ = setup_alchemical_system_and_energy_function(
        name=name, ANImodel=AlchemicalANI2x, env="vacuum"
    )

    AlchemicalANI2x_energy_lambda_1 = energy_function.calculate_energy(
        coordinates, lambda_value=1.0
    )

    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name, ANImodel=CompartimentedAlchemicalANI2x, env="vacuum"
    )

    CompartimentedAlchemicalANI2x_1 = energy_function.calculate_energy(
        coordinates, lambda_value=1.0
    )

    for e1, e2 in zip(
        AlchemicalANI2x_energy_lambda_1.energy, CompartimentedAlchemicalANI2x_1.energy
    ):
        assert is_quantity_close(
            e1.in_units_of(unit.kilojoule_per_mole),
            e2.in_units_of(unit.kilojoule_per_mole),
            rtol=1e-5,
        )


def test_neutromeratio_energy_calculations_with_AlchemicalANI1ccx_in_vacuum():
    from ..tautomers import Tautomer
    import numpy as np
    from ..constants import kT
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, ANI1ccx

    # read in exp_results.pickle
    with open("data/test_data/exp_results.pickle", "rb") as f:
        exp_results = pickle.load(f)

    ######################################################################
    # vacuum
    ######################################################################
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

    energy_1 = energy_function.calculate_energy(x0, lambda_value=1.0)
    for e1, e2 in zip(
        energy_1.energy,
        [
            -906555.29945346,
            -905750.20471091,
            -906317.24952004,
            -906545.17543265,
            -906581.65215098,
            -906618.2832786,
            -906565.05631782,
            -905981.82167316,
            -904681.20632002,
            -904296.8214631,
        ]
        * unit.kilojoule_per_mole,
    ):
        assert is_quantity_close(e1, e2, rtol=1e-2)

    energy_0 = energy_function.calculate_energy(x0, lambda_value=0.0)
    assert is_quantity_close(
        energy_0.energy[0].in_units_of(unit.kilojoule_per_mole),
        (-906912.01647632 * unit.kilojoule_per_mole),
        rtol=1e-9,
    )
    ######################################################################
    # compare with ANI1ccx
    traj, _ = _get_traj(traj_path, top_path, [tautomer.hybrid_hydrogen_idx_at_lambda_1])
    model = ANI1ccx()

    energy_function = neutromeratio.ANI_force_and_energy(
        model=model, atoms=tautomer.final_state_ligand_atoms, mol=None
    )

    coordinates = [x.xyz[0] for x in traj[0]] * unit.nanometer
    assert len(tautomer.initial_state_ligand_atoms) == len(coordinates[0])
    assert is_quantity_close(
        energy_0.energy[0], energy_function.calculate_energy(coordinates).energy
    )

    traj, _ = _get_traj(traj_path, top_path, [tautomer.hybrid_hydrogen_idx_at_lambda_0])

    energy_function = neutromeratio.ANI_force_and_energy(
        model=model, atoms=tautomer.initial_state_ligand_atoms, mol=None
    )

    coordinates = [x.xyz[0] for x in traj[0]] * unit.nanometer
    assert len(tautomer.final_state_ligand_atoms) == len(coordinates[0])
    assert is_quantity_close(
        energy_1.energy[0], energy_function.calculate_energy(coordinates).energy
    )


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_neutromeratio_energy_calculations_with_AlchemicalANI1ccx_in_droplet():
    from ..tautomers import Tautomer
    import numpy as np
    from ..constants import kT
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, ANI1ccx

    # read in exp_results.pickle
    with open("data/test_data/exp_results.pickle", "rb") as f:
        exp_results = pickle.load(f)

    ######################################################################
    # droplet
    ######################################################################
    name = "molDWRow_298"
    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name,
        ANImodel=AlchemicalANI1ccx,
        env="droplet",
        base_path="data/test_data/droplet/molDWRow_298/",
        diameter=10,
    )

    # read in pregenerated traj
    traj_path = (
        "data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
    )
    top_path = "data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    traj, _ = _get_traj(traj_path, top_path)

    # overwrite the coordinates that rdkit generated with the first frame in the traj
    x0 = [x.xyz[0] for x in traj[:10]] * unit.nanometer
    energy_1 = energy_function.calculate_energy(x0, lambda_value=1.0)
    energy_0 = energy_function.calculate_energy(x0, lambda_value=0.0)

    assert len(tautomer.ligand_in_water_atoms) == len(x0[0])
    print(energy_1.energy.in_units_of(unit.kilojoule_per_mole))

    print(energy_1.energy)

    for e1, e2 in zip(
        energy_1.energy,
        [
            -3514015.25593752,
            -3513765.22323459,
            -3512655.23621176,
            -3512175.08728504,
            -3512434.17610022,
            -3513923.68325093,
            -3513997.76968092,
            -3513949.58322023,
            -3513957.74678051,
            -3514045.84769267,
        ]
        * unit.kilojoule_per_mole,
    ):
        assert is_quantity_close(e1, e2, rtol=1e-5)
    for e1, e2 in zip(
        energy_0.energy,
        [
            -3514410.82062014,
            -3514352.85161146,
            -3514328.06891661,
            -3514324.15896465,
            -3514323.94519662,
            -3514326.6575155,
            -3514320.69798817,
            -3514326.79413299,
            -3514309.49535377,
            -3514308.0598529,
        ]
        * unit.kilojoule_per_mole,
    ):
        assert is_quantity_close(e1, e2, rtol=1e-5)

    ######################################################################
    # compare with ANI1ccx -- test1
    # this test removes all restraints
    name = "molDWRow_298"
    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name,
        ANImodel=AlchemicalANI1ccx,
        env="droplet",
        base_path="data/test_data/droplet/molDWRow_298/",
        diameter=10,
    )
    # remove restraints
    energy_function.list_of_lambda_restraints = []

    # overwrite the coordinates that rdkit generated with the first frame in the traj
    x0 = [x.xyz[0] for x in traj[:10]] * unit.nanometer
    energy_1 = energy_function.calculate_energy(x0, lambda_value=1.0)
    energy_0 = energy_function.calculate_energy(x0, lambda_value=0.0)

    model = ANI1ccx()

    traj, _ = _get_traj(traj_path, top_path, [tautomer.hybrid_hydrogen_idx_at_lambda_1])
    atoms = (
        tautomer.ligand_in_water_atoms[: tautomer.hybrid_hydrogen_idx_at_lambda_1]
        + tautomer.ligand_in_water_atoms[tautomer.hybrid_hydrogen_idx_at_lambda_1 + 1 :]
    )

    energy_function = neutromeratio.ANI_force_and_energy(
        model=model, atoms=atoms, mol=None
    )

    coordinates = [x.xyz[0] for x in traj[:10]] * unit.nanometer
    assert len(atoms) == len(coordinates[0])

    energies_ani1ccx_0 = energy_function.calculate_energy(coordinates)
    assert is_quantity_close(
        energy_0.energy[0].in_units_of(unit.kilojoule_per_mole),
        energies_ani1ccx_0.energy[0].in_units_of(unit.kilojoule_per_mole),
    )

    ######################################################################
    # compare with ANI1ccx -- test2
    # includes restraint energy
    name = "molDWRow_298"
    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
        name=name,
        ANImodel=AlchemicalANI1ccx,
        env="droplet",
        base_path="data/test_data/droplet/molDWRow_298/",
        diameter=10,
    )

    # read in pregenerated traj
    traj_path = (
        "data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
    )
    top_path = "data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
    traj, _ = _get_traj(traj_path, top_path)

    x0 = [x.xyz[0] for x in traj[:10]] * unit.nanometer
    energy_1 = energy_function.calculate_energy(x0, lambda_value=1.0)
    energy_0 = energy_function.calculate_energy(x0, lambda_value=0.0)

    model = ANI1ccx()
    traj, _ = _get_traj(traj_path, top_path, [tautomer.hybrid_hydrogen_idx_at_lambda_1])
    atoms = (
        tautomer.ligand_in_water_atoms[: tautomer.hybrid_hydrogen_idx_at_lambda_1]
        + tautomer.ligand_in_water_atoms[tautomer.hybrid_hydrogen_idx_at_lambda_1 + 1 :]
    )

    energy_function = neutromeratio.ANI_force_and_energy(
        model=model, atoms=atoms, mol=None
    )

    coordinates = [x.xyz[0] for x in traj[:10]] * unit.nanometer
    assert len(atoms) == len(coordinates[0])

    energies_ani1ccx_0 = energy_function.calculate_energy(coordinates)
    # get restraint energy
    energy_0_restraint = energy_0.restraint_energy_contribution.in_units_of(
        unit.kilojoule_per_mole
    )
    print(energy_0_restraint[0])
    print(energy_0.energy[0])
    print(energy_0)
    # subtracting restraint energies
    energy_0_minus_restraint = energy_0.energy[0] - energy_0_restraint[0]
    assert is_quantity_close(
        energy_0_minus_restraint,
        energies_ani1ccx_0.energy[0].in_units_of(unit.kilojoule_per_mole),
    )
