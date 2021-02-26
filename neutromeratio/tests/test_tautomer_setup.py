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
from neutromeratio.constants import device
from openmmtools.utils import is_quantity_close
from rdkit import Chem


def _remove_files(name, max_epochs=1):
    try:
        os.remove(f"{name}.pt")
    except FileNotFoundError:
        pass
    for i in range(1, max_epochs):
        os.remove(f"{name}_{i}.pt")
    os.remove(f"{name}_best.pt")


def test_equ():
    assert 1.0 == 1.0


def test_neutromeratio_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "neutromeratio" in sys.modules


def test_tautomer_class():

    from neutromeratio.tautomers import Tautomer

    print(os.getcwd())
    with open("data/test_data/exp_results.pickle", "rb") as f:
        exp_results = pickle.load(f)

    name = "molDWRow_298"
    t1_smiles = exp_results[name]["t1-smiles"]
    t2_smiles = exp_results[name]["t2-smiles"]

    from_mol_tautomer_idx = 1
    to_mol_tautomer_idx = 2

    # generate both rdkit mol
    mols = {
        "t1": neutromeratio.generate_rdkit_mol(t1_smiles),
        "t2": neutromeratio.generate_rdkit_mol(t2_smiles),
    }
    from_mol = mols[f"t{from_mol_tautomer_idx}"]
    to_mol = mols[f"t{to_mol_tautomer_idx}"]
    t = Tautomer(name=name, initial_state_mol=from_mol, final_state_mol=to_mol)
    t.perform_tautomer_transformation()


def test_tautomer_transformation():
    from neutromeratio.tautomers import Tautomer

    with open("data/test_data/exp_results.pickle", "rb") as f:
        exp_results = pickle.load(f)

    ###################################
    ###################################
    name = "molDWRow_298"
    t1_smiles = exp_results[name]["t1-smiles"]
    t2_smiles = exp_results[name]["t2-smiles"]

    (
        t_type,
        tautomers,
        flipped,
    ) = neutromeratio.utils.generate_tautomer_class_stereobond_aware(
        name, t1_smiles, t2_smiles
    )
    t = tautomers[0]
    t.perform_tautomer_transformation()
    assert len(tautomers) == 2
    assert (
        neutromeratio.utils.get_stereotag_of_stereobonds(t.initial_state_mol)
        == "STEREOZ"
    )

    # test if t1 centric mapping of hydrogen, heavy atom acceptor and donor works
    assert t.initial_state_ligand_atoms == "CCCCCOOHHHHHHHH"
    assert t.final_state_ligand_atoms == "CCCCCOOHHHHHHHH"

    assert t.hybrid_hydrogen_idx_at_lambda_0 == 14
    assert t.heavy_atom_hydrogen_acceptor_idx == 2
    assert t.heavy_atom_hydrogen_donor_idx == 5

    # test if dual topology hybrid works
    assert t.hybrid_atoms == "CCCCCOOHHHHHHHHH"
    assert t.hybrid_hydrogen_idx_at_lambda_0 == 14
    assert t.hybrid_hydrogen_idx_at_lambda_1 == 15

    t = tautomers[1]
    t.perform_tautomer_transformation()
    assert (
        neutromeratio.utils.get_stereotag_of_stereobonds(t.initial_state_mol)
        == "STEREOE"
    )

    # test if t1 centric mapping of hydrogen, heavy atom acceptor and donor works
    assert t.initial_state_ligand_atoms == "CCOCCCOHHHHHHHH"
    assert t.hybrid_hydrogen_idx_at_lambda_0 == 14
    assert t.heavy_atom_hydrogen_acceptor_idx == 3
    assert t.heavy_atom_hydrogen_donor_idx == 6

    # test if dual topology hybrid works
    assert t.hybrid_atoms == "CCOCCCOHHHHHHHHH"
    assert t.hybrid_hydrogen_idx_at_lambda_0 == 14
    assert t.hybrid_hydrogen_idx_at_lambda_1 == 15

    ###################################
    ###################################

    name = "molDWRow_37"
    t1_smiles = exp_results[name]["t1-smiles"]
    t2_smiles = exp_results[name]["t2-smiles"]

    (
        t_type,
        tautomers,
        flipped,
    ) = neutromeratio.utils.generate_tautomer_class_stereobond_aware(
        name, t1_smiles, t2_smiles
    )
    assert len(tautomers) == 2
    t = tautomers[0]
    t.perform_tautomer_transformation()
    assert (
        neutromeratio.utils.get_stereotag_of_stereobonds(t.initial_state_mol)
        == "STEREOZ"
    )

    # test if t1 centric mapping of hydrogen, heavy atom acceptor and donor works
    assert t.initial_state_ligand_atoms == "CCOCCCCCOHHHHHHHHHH"
    assert t.hydrogen_idx == 12
    assert t.heavy_atom_hydrogen_acceptor_idx == 8
    assert t.heavy_atom_hydrogen_donor_idx == 2

    # test if dual topology hybrid works
    assert t.hybrid_atoms == "CCOCCCCCOHHHHHHHHHHH"
    assert t.hybrid_hydrogen_idx_at_lambda_0 == 12
    assert t.hybrid_hydrogen_idx_at_lambda_1 == 19

    t = tautomers[1]
    t.perform_tautomer_transformation()
    assert (
        neutromeratio.utils.get_stereotag_of_stereobonds(t.initial_state_mol)
        == "STEREOE"
    )

    # test if t1 centric mapping of hydrogen, heavy atom acceptor and donor works
    assert t.initial_state_ligand_atoms == "CCOCCCCCOHHHHHHHHHH"
    assert t.hydrogen_idx == 12
    assert t.heavy_atom_hydrogen_acceptor_idx == 8
    assert t.heavy_atom_hydrogen_donor_idx == 2

    # test if dual topology hybrid works
    assert t.hybrid_atoms == "CCOCCCCCOHHHHHHHHHHH"
    assert t.hybrid_hydrogen_idx_at_lambda_0 == 12
    assert t.hybrid_hydrogen_idx_at_lambda_1 == 19

    # test if droplet works
    t.add_droplet(t.final_state_ligand_topology, t.get_final_state_ligand_coords(0))

    name = "molDWRow_1233"
    t1_smiles = exp_results[name]["t1-smiles"]
    t2_smiles = exp_results[name]["t2-smiles"]

    (
        t_type,
        tautomers,
        flipped,
    ) = neutromeratio.utils.generate_tautomer_class_stereobond_aware(
        name, t1_smiles, t2_smiles
    )
    assert len(tautomers) == 2
    t = tautomers[0]
    t.perform_tautomer_transformation()
    assert (
        neutromeratio.utils.get_stereotag_of_stereobonds(t.initial_state_mol)
        == "STEREOZ"
    )

    # test if t1 centric mapping of hydrogen, heavy atom acceptor and donor works
    assert t.initial_state_ligand_atoms == "NCOCCCCCCCNNHHHHHHH"
    assert t.hydrogen_idx == 18
    assert t.heavy_atom_hydrogen_acceptor_idx == 0
    assert t.heavy_atom_hydrogen_donor_idx == 11

    # test if dual topology hybrid works
    assert t.hybrid_atoms == "NCOCCCCCCCNNHHHHHHHH"
    assert t.hybrid_hydrogen_idx_at_lambda_0 == 18
    assert t.hybrid_hydrogen_idx_at_lambda_1 == 19

    t = tautomers[1]
    t.perform_tautomer_transformation()
    assert (
        neutromeratio.utils.get_stereotag_of_stereobonds(t.initial_state_mol)
        == "STEREOE"
    )

    # test if t1 centric mapping of hydrogen, heavy atom acceptor and donor works
    assert t.initial_state_ligand_atoms == "NCOCCCCCCCNNHHHHHHH"
    assert t.hydrogen_idx == 18
    assert t.heavy_atom_hydrogen_acceptor_idx == 0
    assert t.heavy_atom_hydrogen_donor_idx == 11

    # test if dual topology hybrid works
    assert t.hybrid_atoms == "NCOCCCCCCCNNHHHHHHHH"
    assert t.hybrid_hydrogen_idx_at_lambda_0 == 18
    assert t.hybrid_hydrogen_idx_at_lambda_1 == 19

    # test if droplet works
    t.add_droplet(t.final_state_ligand_topology, t.get_final_state_ligand_coords(0))


def test_tautomer_transformation_for_all_systems():
    from neutromeratio.tautomers import Tautomer
    from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds
    from ..constants import _get_names
    import random

    with open("data/test_data/exp_results.pickle", "rb") as f:
        exp_results = pickle.load(f)

    name_list = _get_names()
    random.shuffle(name_list)
    ###################################
    ###################################
    for name in name_list[:10]:
        if name in exclude_set_ANI + mols_with_charge + multiple_stereobonds:
            continue
        t1_smiles = exp_results[name]["t1-smiles"]
        t2_smiles = exp_results[name]["t2-smiles"]

        (
            t_type,
            tautomers,
            flipped,
        ) = neutromeratio.utils.generate_tautomer_class_stereobond_aware(
            name, t1_smiles, t2_smiles
        )
        t = tautomers[0]
        t.perform_tautomer_transformation()


def test_species_conversion():
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x, ANI1x, ANI2x, ANI1ccx
    import random, shutil
    import parmed as pm
    import numpy as np
    from ..constants import _get_names

    with open("data/test_data/exp_results.pickle", "rb") as f:
        exp_results = pickle.load(f)

    name_list = _get_names()
    ###################################
    ###################################
    name = name_list[10]
    t1_smiles = exp_results[name]["t1-smiles"]
    t2_smiles = exp_results[name]["t2-smiles"]

    (
        t_type,
        tautomers,
        flipped,
    ) = neutromeratio.utils.generate_tautomer_class_stereobond_aware(
        name, t1_smiles, t2_smiles
    )
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    for model in [ANI1x, ANI2x, ANI1ccx]:
        m = model()
        energy_function = neutromeratio.ANI_force_and_energy(
            model=m, atoms=tautomer.initial_state_ligand_atoms, mol=None
        )

        t = m.species_to_tensor(tautomer.initial_state_ligand_atoms)
        assert torch.all(t.eq(torch.tensor([3, 1, 2, 2, 1, 1, 1, 3, 0, 0, 0, 0])))
        print(m.species_to_tensor(tautomer.initial_state_ligand_atoms))

    for model in [AlchemicalANI2x, AlchemicalANI1ccx]:
        m = model([1, 2])
        energy_function = neutromeratio.ANI_force_and_energy(
            model=m, atoms=tautomer.initial_state_ligand_atoms, mol=None
        )
        assert torch.all(t.eq(torch.tensor([3, 1, 2, 2, 1, 1, 1, 3, 0, 0, 0, 0])))

        print(m.species_to_tensor(tautomer.initial_state_ligand_atoms))


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
        (
            energy_function,
            tautomer,
            flipped,
        ) = setup_alchemical_system_and_energy_function(
            name=name, env="vacuum", ANImodel=AlchemicalANI1ccx
        )
        x0 = tautomer.get_hybrid_coordinates()
        f = energy_function.calculate_force(x0, lambda_value)
        f = energy_function.calculate_energy(x0, lambda_value)

    lambda_value = 0.1
    random.shuffle(names)
    for name in names[:10]:
        (
            energy_function,
            tautomer,
            flipped,
        ) = setup_alchemical_system_and_energy_function(
            name=name, env="vacuum", ANImodel=AlchemicalANI2x
        )
        x0 = tautomer.get_hybrid_coordinates()
        f = energy_function.calculate_force(x0, lambda_value)
        f = energy_function.calculate_energy(x0, lambda_value)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
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
            (
                energy_function,
                tautomer,
                flipped,
            ) = setup_alchemical_system_and_energy_function(
                name=name,
                env="droplet",
                ANImodel=AlchemicalANI1ccx,
                base_path="pdbs-ani1ccx",
                diameter=10,
            )
            x0 = tautomer.get_ligand_in_water_coordinates()
            energy_function.calculate_force(x0, lambda_value)
            energy_function.calculate_energy(x0, lambda_value)

    finally:
        shutil.rmtree("pdbs-ani1ccx")

    lambda_value = 0.1
    random.shuffle(names)
    try:
        for name in names[:10]:
            print(name)
            (
                energy_function,
                tautomer,
                flipped,
            ) = setup_alchemical_system_and_energy_function(
                name=name,
                env="droplet",
                ANImodel=AlchemicalANI2x,
                base_path="pdbs-ani2x",
                diameter=10,
            )
            x0 = tautomer.get_ligand_in_water_coordinates()
            energy_function.calculate_force(x0, lambda_value)
            energy_function.calculate_energy(x0, lambda_value)

    finally:
        shutil.rmtree("pdbs-ani2x")


def test_query_names():
    from ..utils import find_idx

    f = find_idx(query_name="molDWRow_954")
    print(f)

    assert len(f[0]) == 11
    assert f[0][0] == 3686
    assert f[-1] == 336

    l = []
    for n in [
        "molDWRow_1366",
        "molDWRow_1367",
        "molDWRow_1368",
        "molDWRow_1370",
        "molDWRow_1377",
        "molDWRow_1378",
        "molDWRow_1379",
        "molDWRow_1380",
        "molDWRow_1652",
        "molDWRow_1653",
        "molDWRow_1654",
        "molDWRow_1655",
        "molDWRow_1656",
        "molDWRow_1657",
        "molDWRow_1658",
        "molDWRow_1659",
        "molDWRow_575",
        "molDWRow_578",
        "molDWRow_579",
        "molDWRow_58",
        "molDWRow_583",
        "molDWRow_584",
        "molDWRow_59",
        "molDWRow_590",
        "molDWRow_591",
        "molDWRow_592",
        "molDWRow_593",
        "molDWRow_594",
        "molDWRow_595",
        "molDWRow_596",
        "molDWRow_597",
        "molDWRow_598",
        "molDWRow_599",
        "molDWRow_600",
    ]:
        l.append(find_idx(query_name=n)[-1])

    assert l == [93, 94, 95, 96, 97, 98, 99, 100, 193, 194, 195, 196, 197, 198, 199, 200, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300]

    f = find_idx(query_name="molDWRow_575")
    print(f)




@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_setup_tautomer_system_in_droplet_for_problem_systems():
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..ani import AlchemicalANI2x
    import shutil

    # NOTE: Sometimes this test fails? something wrong with molDWRow_68?
    from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds

    try:
        shutil.rmtree("pdbs-ani2ccx-problems")
    except OSError:
        pass

    names = [
        "SAMPLmol2",
        "molDWRow_507",
        "molDWRow_1598",
        "molDWRow_511",
        "molDWRow_516",
        "molDWRow_68",
        "molDWRow_735",
        "molDWRow_895",
    ] + [
        "molDWRow_512",
        "molDWRow_126",
        "molDWRow_554",
        "molDWRow_601",
        "molDWRow_614",
        "molDWRow_853",
        "molDWRow_602",
    ]

    lambda_value = 0.1
    for name in names:
        print(name)
        (
            energy_function,
            tautomer,
            flipped,
        ) = setup_alchemical_system_and_energy_function(
            name=name,
            env="droplet",
            ANImodel=AlchemicalANI2x,
            base_path="pdbs-ani2ccx-problems",
            diameter=10,
        )
        x0 = tautomer.get_ligand_in_water_coordinates()
        energy_function.calculate_force(x0, lambda_value)
        energy_function.calculate_energy(x0, lambda_value)


def test_setup_tautomer_system_in_droplet_with_pdbs():
    from ..analysis import setup_alchemical_system_and_energy_function
    from ..constants import exclude_set_ANI, mols_with_charge, multiple_stereobonds
    import random, shutil
    from ..ani import AlchemicalANI1ccx, AlchemicalANI2x
    from ..constants import _get_names

    names = _get_names()
    random.shuffle(names)

    lambda_value = 0.0
    for name in names[:10]:
        print(name)
        (
            energy_function,
            tautomer,
            flipped,
        ) = setup_alchemical_system_and_energy_function(
            name=name,
            env="droplet",
            ANImodel=AlchemicalANI2x,
            base_path=f"data/test_data/droplet_test/{name}",
            diameter=10,
        )
        x0 = tautomer.get_ligand_in_water_coordinates()
        energy_function.calculate_force(x0, lambda_value)


def test_min_and_single_point_energy():

    from ..ani import ANI1ccx

    # name of the system
    name = "molDWRow_298"

    # extract smiles
    exp_results = pickle.load(open("data/test_data/exp_results.pickle", "rb"))
    t1_smiles = exp_results[name]["t1-smiles"]
    t2_smiles = exp_results[name]["t2-smiles"]

    nr_of_confs = 10
    (
        t_type,
        tautomers,
        flipped,
    ) = neutromeratio.utils.generate_tautomer_class_stereobond_aware(
        name=name,
        t1_smiles=t1_smiles,
        t2_smiles=t2_smiles,
        nr_of_conformations=nr_of_confs,
    )
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    # set model
    model = ANI1ccx()
    model = model.to(device)
    torch.set_num_threads(1)

    # calculate energy using both structures and pure ANI1ccx
    for ase_mol, ligand_atoms, ligand_coords in zip(
        [tautomer.initial_state_ase_mol, tautomer.final_state_ase_mol],
        [tautomer.initial_state_ligand_atoms, tautomer.final_state_ligand_atoms],
        [
            tautomer.get_initial_state_ligand_coords,
            tautomer.get_final_state_ligand_coords,
        ],
    ):
        energy_function = neutromeratio.ANI_force_and_energy(
            model=model, atoms=ligand_atoms, mol=ase_mol
        )

        for i in range(nr_of_confs):
            # minimize
            x0, hist_e = energy_function.minimize(ligand_coords(i))
            print(energy_function.calculate_energy(x0).energy)


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Slow tests fail on travis."
)
def test_thermochemistry():
    from ..ani import ANI1ccx

    # name of the system
    name = "molDWRow_298"

    # extract smiles
    exp_results = pickle.load(open("data/test_data/exp_results.pickle", "rb"))
    t1_smiles = exp_results[name]["t1-smiles"]
    t2_smiles = exp_results[name]["t2-smiles"]

    nr_of_confs = 10
    (
        t_type,
        tautomers,
        flipped,
    ) = neutromeratio.utils.generate_tautomer_class_stereobond_aware(
        name=name,
        t1_smiles=t1_smiles,
        t2_smiles=t2_smiles,
        nr_of_conformations=nr_of_confs,
    )
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    # set model
    model = ANI1ccx()
    model = model.to(device)
    torch.set_num_threads(1)

    # calculate energy using both structures and pure ANI1ccx
    for ase_mol, ligand_atoms, ligand_coords in zip(
        [tautomer.initial_state_ase_mol, tautomer.final_state_ase_mol],
        [tautomer.initial_state_ligand_atoms, tautomer.final_state_ligand_atoms],
        [
            tautomer.get_initial_state_ligand_coords,
            tautomer.get_final_state_ligand_coords,
        ],
    ):
        energy_function = neutromeratio.ANI_force_and_energy(
            model=model,
            atoms=ligand_atoms,
            mol=ase_mol,
        )
        for i in range(nr_of_confs):
            # minimize
            x0, hist_e = energy_function.minimize(ligand_coords(i))
            print(x0.shape)
            energy_function.get_thermo_correction(
                x0
            )  # x has [1][K][3] dimenstion -- N: number of mols, K: number of atoms


def test_change_stereobond():
    from ..utils import change_only_stereobond, get_nr_of_stereobonds

    def get_all_stereotags(smiles):
        mol = Chem.MolFromSmiles(smiles)
        stereo = set()
        for bond in mol.GetBonds():
            if str(bond.GetStereo()) == "STEREONONE":
                continue
            else:
                stereo.add(bond.GetStereo())
        return stereo

    name = "molDWRow_298"
    exp_results = pickle.load(open("data/test_data/exp_results.pickle", "rb"))

    # no stereobond
    smiles = exp_results[name]["t1-smiles"]
    assert get_nr_of_stereobonds(smiles) == 0
    stereo1 = get_all_stereotags(smiles)
    stereo2 = get_all_stereotags(change_only_stereobond(smiles))
    assert stereo1 == stereo2

    # one stereobond
    smiles = exp_results[name]["t2-smiles"]
    assert get_nr_of_stereobonds(smiles) == 1
    stereo1 = get_all_stereotags(smiles)
    stereo2 = get_all_stereotags(change_only_stereobond(smiles))
    assert stereo1 != stereo2


@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Psi4 import fails on travis."
)
def test_tautomer_conformation():
    from ..ani import ANI1ccx

    # name of the system
    name = "molDWRow_298"
    # number of steps

    exp_results = pickle.load(open("data/test_data/exp_results.pickle", "rb"))

    t1_smiles = exp_results[name]["t1-smiles"]
    t2_smiles = exp_results[name]["t2-smiles"]

    (
        t_type,
        tautomers,
        flipped,
    ) = neutromeratio.utils.generate_tautomer_class_stereobond_aware(
        name, t1_smiles, t2_smiles, nr_of_conformations=5
    )
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    print(
        f"Nr of initial conformations: {tautomer.get_nr_of_initial_state_ligand_coords()}"
    )
    print(
        f"Nr of final conformations: {tautomer.get_nr_of_final_state_ligand_coords()}"
    )

    assert tautomer.get_nr_of_initial_state_ligand_coords() == 5
    assert tautomer.get_nr_of_final_state_ligand_coords() == 5

    # set model
    model = ANI1ccx()
    model = model.to(device)
    torch.set_num_threads(1)

    # calculate energy using both structures and pure ANI1ccx
    for ase_mol, ligand_atoms, get_ligand_coords in zip(
        [tautomer.initial_state_ase_mol, tautomer.final_state_ase_mol],
        [tautomer.initial_state_ligand_atoms, tautomer.final_state_ligand_atoms],
        [
            tautomer.get_initial_state_ligand_coords,
            tautomer.get_final_state_ligand_coords,
        ],
    ):
        energy_function = neutromeratio.ANI_force_and_energy(
            model=model,
            atoms=ligand_atoms,
            mol=ase_mol,
        )

        for conf_id in range(5):
            # minimize
            print(f"Conf: {conf_id}")
            x, e_min_history = energy_function.minimize(
                get_ligand_coords(conf_id), maxiter=100000
            )
            energy = energy_function.calculate_energy(
                x
            )  # coordinates need to be in [N][K][3] format
            e_correction = energy_function.get_thermo_correction(x)
            print(f"Energy: {energy.energy}")
            print(f"Energy correction: {e_correction}")

    del model
