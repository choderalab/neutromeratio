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


def test_break_up_calculation():

    """
    Setting / background

    We would like to evaluate $U(x, \theta)$ on a collection of snapshots $x$ that
    doesn't change from iteration to iteration. Further, $\theta$ is the subset of
    model parameters that governs only the final layer in a neural network inside
    the definition of $U$. That means that there is a lot of computation that will
    be identical across all iterations.

    Namely, we can write $U(x, \theta) = g(f(x), \theta)$, where $f$ doesn't depend
    on the parameters $\theta$ we are varying, and $g$ is hopefully cheaper than
    $U$. More specifically, we'd like to define $f(x)$ to be all of the steps up
    until the last layer of the neural network (computing symmetry functions, then
    computing all layers up until the last one).
    """

    from copy import deepcopy
    from neutromeratio.constants import hartree_to_kcal_mol

    import torch
    import numpy as np
    from simtk import unit

    import torch.autograd.profiler as profiler
    import torchani

    print("TorchANI version: ", torchani.__version__)

    from typing import Optional, Tuple

    from torch import Tensor
    from torchani.nn import ANIModel, SpeciesEnergies

    torch.set_num_threads(1)

    class PartialANIModel(ANIModel):
        """just like ANIModel, but don't do the sum over atoms in the last step, and
        don't flatten last layer output!"""

        def forward(
            self,
            species_aev: Tuple[Tensor, Tensor],  # type: ignore
            cell: Optional[Tensor] = None,
            pbc: Optional[Tensor] = None,
        ) -> Tuple[Tensor, Tensor]:

            species, aev = species_aev
            assert species.shape == aev.shape[:-1]

            # in our case, species will be the same for all snapshots
            atom_species = species[0]
            assert (atom_species == species).all()

            # NOTE: depending on the element, outputs will have different dimensions...
            # something like output.shape = n_snapshots, n_atoms, n_dims
            # where n_dims is either 160, 128, or 96...

            # Ugly hard-coding approach: make this of size max_dim=160 and only write
            # into the first 96, 128, or 160 elements, NaN-poisoning the rest
            # TODO: make this less hard-code-y
            n_snapshots, n_atoms = species.shape
            max_dim = 200
            output = torch.zeros((n_snapshots, n_atoms, max_dim)) * np.nan
            # TODO: note intentional NaN-poisoning here -- not sure if there's a
            #   better way to emulate jagged array

            # loop through atom nets
            for i, (_, module) in enumerate(self.items()):
                mask = atom_species == i
                # look only at the elements that are present in species
                if sum(mask) > 0:
                    # get output for these atoms given the aev for these atoms
                    current_out = module(aev[:, mask, :])
                    # dimenstion of current_out is [nr_of_frames, nr_of_atoms_with_element_i,max_dim]
                    out_dim = current_out.shape[-1]
                    # jagged array
                    output[:, mask, :out_dim] = current_out
                    # final dimenstions are [n_snapshots, n_atoms, max_dim]

            return SpeciesEnergies(species, output)

    class LastLayerANIModel(ANIModel):
        """just like ANIModel, but only does the final calculation and cuts input arrays to the input feature size of the
        different atom nets!"""

        last_two_layer_nr_of_feature: dict = {
            -3: {0: 192, 1: 192, 2: 160, 3: 160},
            -1: {0: 160, 1: 160, 2: 128, 3: 128},
        }

        def __init__(self, modules, index_of_last_layer: int):
            super().__init__(modules)
            self.index_of_last_layer = index_of_last_layer

        def forward(
            self,
            species_aev: Tuple[Tensor, Tensor],
            cell: Optional[Tensor] = None,
            pbc: Optional[Tensor] = None,
        ) -> SpeciesEnergies:
            species, aev = species_aev
            species_ = species.flatten()
            aev = aev.flatten(0, 1)

            output = aev.new_zeros(species_.shape)

            for i, (_, m) in enumerate(self.items()):
                mask = species_ == i
                midx = mask.nonzero().flatten()
                if midx.shape[0] > 0:
                    input_ = aev.index_select(0, midx)
                    input_ = input_[
                        :,
                        : self.last_two_layer_nr_of_feature[self.index_of_last_layer][
                            i
                        ],
                    ]
                    output.masked_scatter_(mask, m(input_).flatten())
            output = output.view_as(species)
            return SpeciesEnergies(species, torch.sum(output, dim=1))

    class PartialANIEnsemble(torch.nn.Module):
        def __init__(self, ani_models):
            super().__init__()
            self.ani_models = ani_models

        def forward(self, species_aev):
            species, _ = species_aev
            output = torch.stack(
                [m(species_aev).energies for m in self.ani_models], dim=2
            )

            return SpeciesEnergies(species, output)

    class Precomputation(torch.nn.Module):
        def __init__(self, model: ANIModel, nr_of_included_layers: int):
            super().__init__()
            assert nr_of_included_layers <= 6
            stages = list(model.children())
            assert len(stages) == 4

            ensemble = stages[2]
            assert type(ensemble) == torchani.nn.Ensemble

            # define new ensemble that does everything from AEV up to the last layer
            modified_ensemble = deepcopy(ensemble)

            # remove last layer
            for e in modified_ensemble:
                for element in e.keys():
                    e[element] = e[element][:nr_of_included_layers]

            ani_models = [PartialANIModel(m.children()) for m in modified_ensemble]
            self.partial_ani_ensemble = PartialANIEnsemble(ani_models)
            self.species_converter = stages[0]
            self.aev = stages[1]

        def forward(self, species_coordinates):
            # x = self.species_converter.forward(species_coordinates)
            x = species_coordinates
            species_y = self.partial_ani_ensemble.forward(self.aev.forward(x))
            return species_y

    class LastLayersComputation(torch.nn.Module):
        def __init__(self, model: ANIModel, index_of_last_layers):
            super().__init__()
            stages = list(model.children())
            assert len(stages) == 4
            assert index_of_last_layers == -1 or index_of_last_layers == -3
            ensemble = stages[2]
            assert type(ensemble) == torchani.nn.Ensemble

            # define new ensemble that does just the last layer of computation
            last_step_ensemble = deepcopy(ensemble)
            for e in last_step_ensemble:
                for element in e.keys():
                    e[element] = e[element][index_of_last_layers:]

            ani_models = [
                LastLayerANIModel(m.children(), index_of_last_layers)
                for m in last_step_ensemble
            ]

            self.last_step_ensemble = torchani.nn.Ensemble(ani_models)
            self.energy_shifter = stages[-1]
            assert type(self.energy_shifter) == torchani.EnergyShifter

        def forward(self, species_y):
            """
            TODO: this should only work for elements where the last layer dimension
                is 160
            """
            # y contains the tensor with dimension [n_snapshots, n_atoms, ensemble, max_dimension_of_atom_net (160)]
            species, y = species_y
            n_nets = len(self.last_step_ensemble)
            energies = torch.zeros(y.shape[0])

            # loop through ensembles
            for i in range(n_nets):
                # get last layer for this ensemble
                m = self.last_step_ensemble[i]

                energies += m.forward((species, y[:, :, i, :])).energies
            return self.energy_shifter.forward((species, energies / n_nets))

    def break_into_two_stages(
        model: ANIModel, split_at: int
    ) -> Tuple[Precomputation, LastLayersComputation]:
        """ANIModel.forward(...) is pretty expensive, and in some cases we might want
        to do a computation where the first stage of the calculation is pretty expensive
        and the subsequent stages are less expensive.

        Break ANIModel up into two stages f and g so that
        ANIModel.forward(x) == g.forward(f.forward(x))

        This is beneficial if we only ever need to recompute and adjust g, not f
        """

        if split_at == 6:
            print("Split at layer 6")
            index_of_last_layers = -1
            nr_of_included_layers = 6
        elif split_at == 4:
            print("Split at layer 4")
            index_of_last_layers = -3
            nr_of_included_layers = 4
        else:
            raise RuntimeError("Either split at layer 4 or 6.")

        f = Precomputation(model, nr_of_included_layers=nr_of_included_layers)
        g = LastLayersComputation(model, index_of_last_layers=index_of_last_layers)
        return f, g

    def get_coordinates_and_species_of_droplet(
        n_snapshots: int,
    ) -> Tuple[(Tensor, Tensor)]:
        from neutromeratio.utils import _get_traj

        traj_path = "../data/test_data/droplet/molDWRow_298/molDWRow_298_lambda_0.0000_in_droplet.dcd"
        top_path = "../data/test_data/droplet/molDWRow_298/molDWRow_298_in_droplet.pdb"
        # _get_traj remove the dummy atom
        traj, top, species = _get_traj(traj_path, top_path)
        # overwrite the coordinates that rdkit generated with the first frame in the traj
        n_snapshots_coordinates = (
            [x.xyz[0] for x in traj[:n_snapshots]] * unit.nanometer
        ).value_in_unit(unit.angstrom)
        n_snapshots_species = torch.stack(
            [species[0]] * n_snapshots
        )  # species is a [1][1] tensor, afterwards it's a [1][nr_of_mols]

        n_snapshots_coordinates = torch.tensor(
            n_snapshots_coordinates, dtype=torch.float64, requires_grad=True
        )
        return n_snapshots_coordinates, n_snapshots_species

    if __name__ == "__main__":
        # download and initialize model
        model = torchani.models.ANI2x(periodic_table_index=False)

        # a bunch of snapshots for a droplet system
        n_snapshots = 100

        coordinates, species = get_coordinates_and_species_of_droplet(n_snapshots)

        methane_coords = [
            [0.03192167, 0.00638559, 0.01301679],
            [-0.83140486, 0.39370209, -0.26395324],
            [-0.66518241, -0.84461308, 0.20759389],
            [0.45554739, 0.54289633, 0.81170881],
            [0.66091919, -0.16799635, -0.91037834],
        ]
        coordinates = torch.tensor([methane_coords * 10] * n_snapshots)
        species = torch.tensor([[1, 0, 0, 0, 3] * 10] * n_snapshots)

        species_coordinates = (species, coordinates)

        # the original potential energy function
        with profiler.profile(record_shapes=True, profile_memory=True) as prof:
            with profiler.record_function("model_inference"):
                species_e_ref = model.forward(species_coordinates)
                # print(species_e_ref.energies * hartree_to_kcal_mol)

        s = prof.self_cpu_time_total / 1000000
        print(f"time to compute energies in batch: {s:.3f} s")

        f, g = break_into_two_stages(model, 6)

        species_y = f.forward(species_coordinates)
        species_e = g.forward(species_y)

        e_residuals = species_e.energies - species_e_ref.energies
        print("energy error: |g(f(x)) - model(x)|: ", torch.norm(e_residuals))

        with profiler.profile(record_shapes=True) as prof_f:
            with profiler.record_function("model_inference"):
                species_y = f.forward(species_coordinates)
        s = prof_f.self_cpu_time_total / 1000000
        print(f"time to precompute up until last layer(s) (f): {s:.3f} s")

        with profiler.profile(record_shapes=True) as prof_g:
            with profiler.record_function("model_inference"):
                species_e = g.forward(species_y)
        s = prof_g.self_cpu_time_total / 1000000
        print(f"time to compute last layer(s) (g): {s:.3f} s")

        # finally, compute gradients w.r.t. last layer only
        g.train()

        # note that species_y requires grad
        species, y = species_y
        # print(y.requires_grad) # >True

        # detach so we don't compute expensive gradients w.r.t. y
        species_y_ = SpeciesEnergies(species, y.detach())
        # print(species_y_[1].requires_grad) # >False

        with profiler.profile(record_shapes=True) as prof_backprop:
            with profiler.record_function("model_inference"):
                L = g.forward(species_y_).energies.sum()
                L.backward()

        s = prof_backprop.self_cpu_time_total / 1000000
        print(f"time to compute derivatives of E w.r.t. last layer: {s:.3f} s")


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
    traj, top = _get_traj(traj_path, top_path, tautomer.hybrid_hydrogen_idx_at_lambda_1)
    coordinates = [x.xyz[0] for x in traj[:10]] * unit.nanometer
    assert len(tautomer.initial_state_ligand_atoms) == len(coordinates[0])
    energy_function = neutromeratio.ANI_force_and_energy(
        model=model, atoms=tautomer.final_state_ligand_atoms, mol=None
    )
    ANI2x_energy_final_state = energy_function.calculate_energy(
        coordinates,
    )
    # initial state
    traj, top = _get_traj(traj_path, top_path, tautomer.hybrid_hydrogen_idx_at_lambda_0)
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
    traj, top = _get_traj(traj_path, top_path, None)
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
    traj, top = _get_traj(traj_path, top_path, None)
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
    traj, top = _get_traj(traj_path, top_path, None)
    coordinates = [x.xyz[0] for x in traj[:100]] * unit.nanometer

    energy_function, tautomer, flipped = setup_alchemical_system_and_energy_function(
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
    traj, top = _get_traj(traj_path, top_path, None)

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
    traj, top = _get_traj(traj_path, top_path, tautomer.hybrid_hydrogen_idx_at_lambda_1)
    model = ANI1ccx()

    energy_function = neutromeratio.ANI_force_and_energy(
        model=model, atoms=tautomer.final_state_ligand_atoms, mol=None
    )

    coordinates = [x.xyz[0] for x in traj[0]] * unit.nanometer
    assert len(tautomer.initial_state_ligand_atoms) == len(coordinates[0])
    assert is_quantity_close(
        energy_0.energy[0], energy_function.calculate_energy(coordinates).energy
    )

    traj, top = _get_traj(traj_path, top_path, tautomer.hybrid_hydrogen_idx_at_lambda_0)

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
    traj, top = _get_traj(traj_path, top_path, None)

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

    traj, top = _get_traj(traj_path, top_path, tautomer.hybrid_hydrogen_idx_at_lambda_1)
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
    traj, top = _get_traj(traj_path, top_path, None)

    x0 = [x.xyz[0] for x in traj[:10]] * unit.nanometer
    energy_1 = energy_function.calculate_energy(x0, lambda_value=1.0)
    energy_0 = energy_function.calculate_energy(x0, lambda_value=0.0)

    model = ANI1ccx()
    traj, top = _get_traj(traj_path, top_path, tautomer.hybrid_hydrogen_idx_at_lambda_1)
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
