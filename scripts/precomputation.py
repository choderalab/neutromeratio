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

torch.set_default_dtype(torch.float64)
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
        max_dim = 160
        output = torch.zeros((n_snapshots, n_atoms, max_dim)) * np.nan
        # TODO: note intentional NaN-poisoning here -- not sure if there's a
        #   better way to emulate jagged array

        for i, (_, module) in enumerate(self.items()):
            mask = atom_species == i

            if sum(mask) > 0:
                current_out = module(aev[:, mask, :])
                out_dim = current_out.shape[-1]
                output[:, mask, :out_dim] = current_out

        return SpeciesEnergies(species, output)


class PartialANIEnsemble(torch.nn.Module):
    def __init__(self, ani_models):
        super().__init__()
        self.ani_models = ani_models

    def forward(self, species_aev):
        species, _ = species_aev
        output = torch.stack([m(species_aev).energies for m in self.ani_models], dim=2)
        return SpeciesEnergies(species, output)


class Precomputation(torch.nn.Module):
    def __init__(self, model: ANIModel):
        super().__init__()
        stages = list(model.children())
        assert len(stages) == 4

        ensemble = stages[2]
        assert type(ensemble) == torchani.nn.Ensemble

        # define new ensemble that does everything from AEV up to the last layer
        modified_ensemble = deepcopy(ensemble)

        for e in modified_ensemble:
            for element in e.keys():
                e[element] = e[element][:6]

        ani_models = [PartialANIModel(m.children()) for m in modified_ensemble]
        self.partial_ani_ensemble = PartialANIEnsemble(ani_models)
        self.species_converter = stages[0]
        self.aev = stages[1]

    def forward(self, species_coordinates):
        x = self.species_converter.forward(species_coordinates)
        species_y = self.partial_ani_ensemble.forward(self.aev.forward(x))
        return species_y


class LastLayerComputation(torch.nn.Module):
    def __init__(self, model: ANIModel):
        super().__init__()
        stages = list(model.children())
        assert len(stages) == 4

        ensemble = stages[2]
        assert type(ensemble) == torchani.nn.Ensemble

        # define new ensemble that does just the last layer of computation
        last_step_ensemble = deepcopy(ensemble)
        for e in last_step_ensemble:
            for element in e.keys():
                e[element] = e[element][-1:]

        self.last_step_ensemble = last_step_ensemble
        self.energy_shifter = stages[-1]
        assert type(self.energy_shifter) == torchani.EnergyShifter

    def forward(self, species_y):
        """
        TODO: this should only work for elements where the last layer dimension
            is 160
        """
        species, y = species_y
        n_nets = len(self.last_step_ensemble)
        energies = torch.zeros(y.shape[0])
        for i in range(n_nets):
            m = self.last_step_ensemble[i]
            energies += m.forward((species, y[:, :, i, :])).energies
        return self.energy_shifter.forward((species, energies / n_nets))


def break_into_two_stages(
    model: ANIModel,
) -> Tuple[Precomputation, LastLayerComputation]:
    """ANIModel.forward(...) is pretty expensive, and in some cases we might want
    to do a computation where the first stage of the calculation is pretty expensive
    and the subsequent stages are less expensive.

    Break ANIModel up into two stages f and g so that
    ANIModel.forward(x) == g.forward(f.forward(x))

    This is beneficial if we only ever need to recompute and adjust g, not f
    """

    f = Precomputation(model)
    g = LastLayerComputation(model)
    return f, g


def get_coordinates_and_species_of_droplet(n_snapshots: int) -> Tuple[(Tensor, Tensor)]:
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
    n_snapshots = 500

    coordinates, species = get_coordinates_and_species_of_droplet(n_snapshots)

    # methane_coords = [
    #     [0.03192167, 0.00638559, 0.01301679],
    #     [-0.83140486, 0.39370209, -0.26395324],
    #     [-0.66518241, -0.84461308, 0.20759389],
    #     [0.45554739, 0.54289633, 0.81170881],
    #     [0.66091919, -0.16799635, -0.91037834],
    # ]
    # coordinates = torch.tensor([methane_coords * 10] * n_snapshots)
    # species = torch.tensor([[1, 0, 0, 0, 0] * 10] * n_snapshots)

    print(species.size())
    species_coordinates = (species, coordinates)

    # the original potential energy function
    with profiler.profile(record_shapes=True, profile_memory=True) as prof:
        with profiler.record_function("model_inference"):
            species_e_ref = model.forward(species_coordinates)
            print(species_e_ref.energies * hartree_to_kcal_mol)

    s = prof.self_cpu_time_total / 1000000
    print(f"time to compute energies in batch: {s:.3f} s")

    f, g = break_into_two_stages(model)

    species_y = f.forward(species_coordinates)
    species_e = g.forward(species_y)

    e_residuals = species_e.energies - species_e_ref.energies
    print("energy error: |g(f(x)) - model(x)|: ", torch.norm(e_residuals))

    with profiler.profile(record_shapes=True) as prof_f:
        with profiler.record_function("model_inference"):
            species_y = f.forward(species_coordinates)
    s = prof_f.self_cpu_time_total / 1000000
    print(f"time to precompute up until last layer (f): {s:.3f} s")

    with profiler.profile(record_shapes=True) as prof_g:
        with profiler.record_function("model_inference"):
            species_e = g.forward(species_y)
    s = prof_g.self_cpu_time_total / 1000000
    print(f"time to compute last layer (g): {s:.3f} s")

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
