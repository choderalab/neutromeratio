import copy
import logging
import os
import random
from typing import NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import simtk
import torch
import torchani
from ase import Atoms
from ase.thermochemistry import IdealGasThermo
from ase.vibrations import Vibrations
from simtk import unit
from torch import Tensor
from torchani.nn import ANIModel, SpeciesEnergies

from .constants import (
    device,
    eV_to_kJ_mol,
    hartree_to_kT,
    kT,
    kT_to_kJ_mol,
    nm_to_angstroms,
    platform,
    pressure,
    temperature,
)
from .restraints import BaseDistanceRestraint

logger = logging.getLogger(__name__)


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

        # Ugly hard-coding approach: make this of size max_dim=200 and only write
        # into the first 96, 128, or 160, 190 elements, NaN-poisoning the rest
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

    last_layers_nr_of_feature: dict = {
        "CompartimentedAlchemicalANI2x": {
            -3: {0: 192, 1: 192, 2: 160, 3: 160},
            -1: {0: 160, 1: 160, 2: 128, 3: 128},
        },
        "CompartimentedAlchemicalANI1ccx": {
            -3: {0: 192, 1: 192, 2: 160, 3: 160},
            -1: {0: 96, 1: 96, 2: 96, 3: 96},
        },
    }

    def __init__(self, modules, index_of_last_layer: int, name: str):
        super().__init__(modules)
        self.index_of_last_layer = index_of_last_layer
        self.name = name

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
                    : self.last_layers_nr_of_feature[self.name][
                        self.index_of_last_layer
                    ][i],
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
        output = torch.stack([m(species_aev).energies for m in self.ani_models], dim=2)

        return SpeciesEnergies(species, output)


class Precomputation(torch.nn.Module):
    def __init__(self, model: ANIModel, nr_of_included_layers: int):
        super().__init__()
        assert nr_of_included_layers <= 6

        ensemble = model[0]
        assert type(ensemble) == torchani.nn.Ensemble

        # define new ensemble that does everything from AEV up to the last layer
        modified_ensemble = copy.deepcopy(ensemble)
        # remove last layer
        for e in modified_ensemble:
            for element in e.keys():
                e[element] = e[element][:nr_of_included_layers]

        ani_models = [PartialANIModel(m.children()) for m in modified_ensemble]
        self.partial_ani_ensemble = PartialANIEnsemble(ani_models)
        self.species_converter = model[1]
        self.aev = model[2]

    def forward(self, species_coordinates):
        # x = self.species_converter.forward(species_coordinates)
        x = species_coordinates
        species_y = self.partial_ani_ensemble.forward(self.aev.forward(x))
        return species_y


class LastLayersComputation(torch.nn.Module):
    def __init__(self, model: ANIModel, index_of_last_layers: int, name: str):
        super().__init__()
        assert len(model) == 2
        assert index_of_last_layers == -1 or index_of_last_layers == -3
        ensemble = model[0]
        assert type(ensemble) == torchani.nn.Ensemble

        # define new ensemble that does just the last layer of computation
        last_step_ensemble = copy.deepcopy(
            ensemble
        )  # NOTE: copy reference to original ensemble!
        for e_original, e_copy in zip(ensemble, last_step_ensemble):
            for element in e_original.keys():
                e_copy[element] = e_original[element][index_of_last_layers:]

        ani_models = [
            LastLayerANIModel(m.children(), index_of_last_layers, name)
            for m in last_step_ensemble
        ]

        self.last_step_ensemble = torchani.nn.Ensemble(ani_models)
        self.energy_shifter = model[1]
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


class DecomposedForce(NamedTuple):
    """Returned by _calculate_force()

    force: unit'd
    energy: unit'd
    restraint_bias: unit'd (energy contribution of all restraints)
    stddev: unit'd
    ensemble_bias: unit'd
    """

    force: unit.Quantity
    energy: unit.Quantity
    restraint_energy_contribution: unit.Quantity


class DecomposedEnergy(NamedTuple):
    """Returned by _calculate_energy()

    energy: unit'd
    restraint_energy_contribution: unit'd
    energy_tensor: in kT
    """

    energy: unit.Quantity
    restraint_energy_contribution: unit.Quantity
    energy_tensor: Tensor


class ANI(torchani.models.BuiltinEnsemble):
    def __init__(self, nn_path, periodic_table_index):
        """
        Scale the contributions of alchemical atoms to the energy.
        """
        super().__init__(*self._from_neurochem_resources(nn_path, periodic_table_index))

    def load_nn_parameters(self, parameter_path: str):

        if os.path.isfile(parameter_path):
            parameters = torch.load(parameter_path)
            try:
                self.optimized_neural_network.load_state_dict(parameters["nn"])
            except KeyError:
                self.optimized_neural_network.load_state_dict(parameters)
        else:
            raise RuntimeError(f"Parameter file {parameter_path} does not exist.")

    def _from_neurochem_resources(self, info_file_path, periodic_table_index):
        (
            consts,
            sae_file,
            ensemble_prefix,
            ensemble_size,
        ) = self._parse_neurochem_resources(info_file_path)

        species_converter = torchani.nn.SpeciesConverter(consts.species)
        aev_computer = torchani.aev.AEVComputer(**consts)
        energy_shifter, sae_dict = torchani.neurochem.load_sae(
            sae_file, return_dict=True
        )
        species_to_tensor = consts.species_to_tensor
        neural_networks = torchani.neurochem.load_model_ensemble(
            consts.species, ensemble_prefix, ensemble_size
        )
        return (
            species_converter,
            aev_computer,
            neural_networks.to(device),
            energy_shifter,
            species_to_tensor,
            consts,
            sae_dict,
            periodic_table_index,
        )

    def forward(self, species_coordinates_lamb):

        if len(species_coordinates_lamb) == 4:
            species, coordinates, lam, original_parameters = species_coordinates_lamb
        elif len(species_coordinates_lamb) == 3:
            species, coordinates, original_parameters = species_coordinates_lamb
        elif len(species_coordinates_lamb) == 2:
            species, coordinates = species_coordinates_lamb
            original_parameters = True
        else:
            raise RuntimeError(f"Too many arguments in {species_coordinates_lamb}")

        if original_parameters:
            logger.debug("Using original neural network parameters.")
            nn = self.original_neural_network
        else:
            nn = self.optimized_neural_network
            logger.debug("Using possibly tweaked neural network parameters.")

        species_coordinates = (species, coordinates)
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=None, pbc=None)
        species_energies = nn(species_aevs)
        return self.energy_shifter(species_energies)


class ANI1x(ANI):

    optimized_neural_network = None
    original_neural_network = None
    name = "ANI1x"

    def __init__(self, periodic_table_index: bool = False):
        info_file = "ani-1x_8x.info"
        super().__init__(info_file, periodic_table_index)
        if ANI1x.optimized_neural_network == None:
            ANI1x.optimized_neural_network = copy.deepcopy(self.neural_networks)
        if ANI1x.original_neural_network == None:
            ANI1x.original_neural_network = copy.deepcopy(self.neural_networks)

    @classmethod
    def _reset_parameters(cls):
        if cls.original_neural_network:
            cls.optimized_neural_network = copy.deepcopy(cls.original_neural_network)
        else:
            logger.info("_reset_parameters called, but nothing to do.")


class ANI1ccx(ANI):

    optimized_neural_network = None
    original_neural_network = None
    name = "ANI1ccx"

    def __init__(self, periodic_table_index: bool = False):
        info_file = "ani-1ccx_8x.info"
        super().__init__(info_file, periodic_table_index)
        if ANI1ccx.optimized_neural_network == None:
            ANI1ccx.optimized_neural_network = copy.deepcopy(self.neural_networks)
        if ANI1ccx.original_neural_network == None:
            ANI1ccx.original_neural_network = copy.deepcopy(self.neural_networks)

    @classmethod
    def _reset_parameters(cls):
        if cls.original_neural_network:
            cls.optimized_neural_network = copy.deepcopy(cls.original_neural_network)
        else:
            logger.info("_reset_parameters called, but nothing to do.")


class ANI2x(ANI):
    optimized_neural_network = None
    original_neural_network = None
    name = "ANI2x"

    def __init__(self, periodic_table_index: bool = False):
        info_file = "ani-2x_8x.info"
        super().__init__(info_file, periodic_table_index)
        if ANI2x.optimized_neural_network == None:
            ANI2x.optimized_neural_network = copy.deepcopy(self.neural_networks)
        if ANI2x.original_neural_network == None:
            ANI2x.original_neural_network = copy.deepcopy(self.neural_networks)

    @classmethod
    def _reset_parameters(cls):
        if cls.original_neural_network:
            cls.optimized_neural_network = copy.deepcopy(cls.original_neural_network)
        else:
            logger.info("_reset_parameters called, but nothing to do.")


class AlchemicalANI_Mixin:
    """
    Makes and AlchemicalANI out of ANI.
    """

    @staticmethod
    def _checks(
        mod_species_0,
        mod_species_1,
        species,
        mod_coordinates_0,
        mod_coordinates_1,
        coordinates,
    ):
        if not (
            mod_species_0.size()[0] == species.size()[0]
            and mod_species_0.size()[1] == species.size()[1] - 1
        ):
            raise RuntimeError(
                f"Something went wrong for mod_species_0. Alchemical atoms: {dummy_atom_0} and {dummy_atom_1}. Species tensor size {mod_species_0.size()} is not equal mod species tensor {mod_species_0.size()}"
            )
        if not (
            mod_species_1.size()[0] == species.size()[0]
            and mod_species_1.size()[1] == species.size()[1] - 1
        ):
            raise RuntimeError(
                f"Something went wrong for mod_species_1. Alchemical atoms: {dummy_atom_0} and {dummy_atom_1}. Species tensor size {mod_species_1.size()} is not equal mod species tensor {mod_species_1.size()}"
            )
        if not (
            mod_coordinates_0.size()[0] == coordinates.size()[0]
            and mod_coordinates_0.size()[1] == coordinates.size()[1] - 1
        ):
            raise RuntimeError(
                f"Something went wrong for mod_coordinates_0. Alchemical atoms: {dummy_atom_0} and {dummy_atom_1}. Coord tensor size {mod_coordinates_0.size()} is not equal mod coord tensor {mod_coordinates_0.size()}"
            )
        if not (
            mod_coordinates_1.size()[0] == coordinates.size()[0]
            and mod_coordinates_1.size()[1] == coordinates.size()[1] - 1
        ):
            raise RuntimeError(
                f"Something went wrong for mod_coordinates_1. Alchemical atoms: {dummy_atom_0} and {dummy_atom_1}. Coord tensor size {mod_coordinates_1.size()} is not equal mod coord tensor {mod_coordinates_1.size()}"
            )

    def _forward(self, nn, mod_species, mod_coordinates):
        _, mod_aevs = self.aev_computer((mod_species, mod_coordinates))
        # neural net output given these modified AEVs
        state = nn((mod_species, mod_aevs))
        return self.energy_shifter((mod_species, state.energies))

    @staticmethod
    def _get_modified_species(species, dummy_atom):
        return torch.cat((species[:, :dummy_atom], species[:, dummy_atom + 1 :]), dim=1)

    @staticmethod
    def _get_modified_coordiantes(coordinates, dummy_atom):
        return torch.cat(
            (coordinates[:, :dummy_atom], coordinates[:, dummy_atom + 1 :]), dim=1
        )

    def forward(self, species_coordinates_lamb):
        """
        Energy and stddev are calculated and linearly interpolated between
        the physical endstates at lambda 0 and lamb 1.
        Parameters
        ----------
        species_coordinates
        Returns
        ----------
        E : float
            energy in hartree
        stddev : float
            energy in hartree
        """

        species, coordinates, lam, original_parameters = species_coordinates_lamb

        if original_parameters:
            logger.debug("Using original neural network parameters.")
            nn = self.original_neural_network
        else:
            nn = self.optimized_neural_network
            logger.debug("Using possibly tweaked neural network parameters.")

        # get new species tensor
        mod_species_0 = self._get_modified_species(species, self.alchemical_atoms[0])
        mod_species_1 = self._get_modified_species(species, self.alchemical_atoms[1])

        # get new coordinate tensor
        mod_coordinates_0 = self._get_modified_coordiantes(
            coordinates, self.alchemical_atoms[0]
        )
        mod_coordinates_1 = self._get_modified_coordiantes(
            coordinates, self.alchemical_atoms[1]
        )

        # perform some checks
        self._checks(
            mod_species_0,
            mod_species_1,
            species,
            mod_coordinates_0,
            mod_coordinates_1,
            coordinates,
        )
        # early exit if at endpoint
        if lam == 0.0:
            _, E_0 = self._forward(nn, mod_species_0, mod_coordinates_0)
            return species, E_0

        # early exit if at endpoint
        elif lam == 1.0:
            _, E_1 = self._forward(nn, mod_species_1, mod_coordinates_1)
            return species, E_1

        else:
            _, E_0 = self._forward(nn, mod_species_0, mod_coordinates_0)
            _, E_1 = self._forward(nn, mod_species_1, mod_coordinates_1)
            E = (lam * E_1) + ((1 - lam) * E_0)
            return species, E


class AlchemicalANI1ccx(AlchemicalANI_Mixin, ANI1ccx):

    name = "AlchemicalANI1ccx"

    def __init__(self, alchemical_atoms: list, periodic_table_index: bool = False):
        """Scale the indirect contributions of alchemical atoms to the energy sum by
        linearly interpolating, for other atom i, between the energy E_i^0 it would compute
        in the _complete absence_ of the alchemical atoms, and the energy E_i^1 it would compute
        in the _presence_ of the alchemical atoms.
        (Also scale direct contributions, as in DirectAlchemicalANI)

        Parameters
        ----------
        alchemical_atoms : list
        """

        assert len(alchemical_atoms) == 2
        super().__init__(periodic_table_index)
        self.alchemical_atoms = alchemical_atoms
        self.neural_networks = None
        assert self.neural_networks == None


class AlchemicalANI1x(AlchemicalANI_Mixin, ANI1x):

    name = "AlchemicalANI1x"

    def __init__(self, alchemical_atoms: list, periodic_table_index: bool = False):
        """Scale the indirect contributions of alchemical atoms to the energy sum by
        linearly interpolating, for other atom i, between the energy E_i^0 it would compute
        in the _complete absence_ of the alchemical atoms, and the energy E_i^1 it would compute
        in the _presence_ of the alchemical atoms.
        (Also scale direct contributions, as in DirectAlchemicalANI)

        Parameters
        ----------
        alchemical_atoms : list
        """

        assert len(alchemical_atoms) == 2
        super().__init__(periodic_table_index)
        self.alchemical_atoms = alchemical_atoms
        self.neural_networks = None
        assert self.neural_networks == None


class AlchemicalANI2x(AlchemicalANI_Mixin, ANI2x):

    name = "AlchemicalANI2x"

    def __init__(self, alchemical_atoms: list, periodic_table_index: bool = False):
        """Scale the indirect contributions of alchemical atoms to the energy sum by
        linearly interpolating, for other atom i, between the energy E_i^0 it would compute
        in the _complete absence_ of the alchemical atoms, and the energy E_i^1 it would compute
        in the _presence_ of the alchemical atoms.
        (Also scale direct contributions, as in DirectAlchemicalANI)

        Parameters
        ----------
        alchemical_atoms : list
        """

        assert len(alchemical_atoms) == 2
        super().__init__(periodic_table_index)
        self.alchemical_atoms: list = alchemical_atoms
        self.neural_networks = None
        assert self.neural_networks == None


class ANI_force_and_energy(object):
    def __init__(self, model: ANI, atoms: str, mol: Atoms = None):
        """
        Performs energy and force calculations.

        Parameters
        ----------
        model: torchani.models
        atoms: str
            a string of atoms in the indexed order
        mol (optional): ase.Atoms
            a ASE Atoms object with the atoms
        """
        self.device = device
        self.model = model
        self.atoms = atoms
        self.ase_mol = mol
        self.species = model.species_to_tensor(atoms).to(device).unsqueeze(0)
        self.platform = platform
        self.list_of_lambda_restraints: list = []

        # TODO: check availablity of platform

    def add_restraint_to_lambda_protocol(self, restraint):
        """
        Add a single restraint to the lambda protocol.

        Arguments:
            restraint {neutromeratio.restraint.Restraint} -- Either Harmonic or FlatBottomRestraint
        """
        assert isinstance(restraint, BaseDistanceRestraint)
        self.list_of_lambda_restraints.append(restraint)

    def reset_lambda_restraints(self):
        """
        Resets the restraints for the lambda protocol
        """
        self.list_of_lambda_restraints = []

    def _compute_restraint_bias(self, coordinates, lambda_value):
        """
        Computes the energy from different restraints of the system.

        Arguments:
            x {Tensor} -- coordinates as torch.Tensor in nanometer
            lambda_value {float} -- lambda value

        Raises:
            RuntimeError: raises RuntimeError if restraint.active_at has numeric value outside [0,1]

        Returns:
            float -- energy [kT]
        """

        # use correct restraint_bias in between the end-points...
        from neutromeratio.constants import kJ_mol_to_kT

        nr_of_mols = len(coordinates)
        restraint_bias_in_kT = torch.tensor(
            [0.0] * nr_of_mols, device=self.device, dtype=torch.float64
        )
        coordinates_in_angstrom = coordinates * nm_to_angstroms
        for restraint in self.list_of_lambda_restraints:
            restraint_bias = restraint.restraint(coordinates_in_angstrom)
            restraint_bias_in_kT += restraint_bias * kJ_mol_to_kT

        return restraint_bias_in_kT

    def get_thermo_correction(
        self, coords: simtk.unit.quantity.Quantity
    ) -> unit.quantity.Quantity:

        """
        Returns the thermochemistry correction. This calls: https://wiki.fysik.dtu.dk/ase/ase/thermochemistry/thermochemistry.html
        and uses the Ideal gas rigid rotor harmonic oscillator approximation to calculate the Gibbs free energy correction that
        needs to be added to the single point energy to obtain the Gibb's free energy
        coords: [K][3]

        Raises:
            verror: if imaginary frequencies are detected a ValueError is raised

        Returns:
            float -- temperature correct [kT]
        """
        if not (
            len(coords.shape) == 3 and coords.shape[2] == 3 and coords.shape[0] == 1
        ):
            raise RuntimeError(
                f"Something is wrong with the shape of the provided coordinates: {coords.shape}. Only x.shape[0] == 1 is possible."
            )

        ase_mol = copy.deepcopy(self.ase_mol)
        for atom, c in zip(ase_mol, coords[0]):
            atom.x = c[0].value_in_unit(unit.angstrom)
            atom.y = c[1].value_in_unit(unit.angstrom)
            atom.z = c[2].value_in_unit(unit.angstrom)

        calculator = self.model.ase()
        ase_mol.set_calculator(calculator)

        vib = Vibrations(ase_mol, name=f"/tmp/vib{random.randint(1,10000000)}")
        vib.run()
        vib_energies = vib.get_energies()
        thermo = IdealGasThermo(
            vib_energies=vib_energies,
            atoms=ase_mol,
            geometry="nonlinear",
            symmetrynumber=1,
            spin=0,
        )

        try:
            G = thermo.get_gibbs_energy(
                temperature=temperature.value_in_unit(unit.kelvin),
                pressure=pressure.value_in_unit(unit.pascal),
            )
        except ValueError as verror:
            logger.critical(verror)
            vib.clean()
            raise verror
        # removes the vib tmp files
        vib.clean()
        return (
            G * eV_to_kJ_mol
        ) * unit.kilojoule_per_mole  # eV * conversion_factor(eV to kJ/mol)

    def minimize(
        self,
        coords: simtk.unit.quantity.Quantity,
        maxiter: int = 1000,
        lambda_value: float = 0.0,
        show_plot: bool = False,
    ) -> Tuple[simtk.unit.quantity.Quantity, list]:
        """Minimizes the molecule

        Arguments:
            coords {simtk.unit.quantity.Quantity} -- coordinates of the molecules unit'd

        Keyword Arguments:
            maxiter {int} -- max iteration performed by minimizer (default: {1000})
            lambda_value {float} -- lambda value (default: {0.0})
            show_plot {bool} -- show summary plot after minimization finshes (default: {False})

        Returns:
            coordinates
            list -- trajectory of energy values during minimization
        """
        from scipy import optimize

        assert type(coords) == unit.Quantity

        def plotting(y1, y2, y1_axis_label, y1_label, y2_axis_label, y2_label, title):
            fig, ax1 = plt.subplots()
            plt.title(title)

            color = "tab:red"
            ax1.set_xlabel("timestep (10 fs)")
            ax1.set_ylabel(y1_axis_label, color=color)
            plt.plot([e / kT for e in y1], label=y1_label, color=color)
            ax1.tick_params(axis="y", labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = "tab:blue"
            ax2.set_ylabel(
                y2_axis_label, color=color
            )  # we already handled the x-label with ax1
            plt.plot([e / kT for e in y2], label=y2_label, color=color)
            ax2.tick_params(axis="y", labelcolor=color)
            ax2.set_xlabel("timestep")
            plt.legend()
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()
            plt.close()

        x = coords.value_in_unit(unit.angstrom)
        if not (len(x.shape) == 3 and x.shape[2] == 3 and x.shape[0] == 1):
            raise RuntimeError(
                f"Something is wrong with the shape of the provided coordinates: {x.shape}. Only x.shape[0] == 1 is possible."
            )
        self.memory_of_energy: list = []
        self.memory_of_restrain_contribution: list = []

        logger.info("Begin minimizing...")
        f = optimize.minimize(
            self._traget_energy_function,
            x,
            method="BFGS",
            jac=True,
            args=(lambda_value),
            options={"maxiter": maxiter, "disp": True},
        )

        logger.critical(f"Minimization status: {f.success}")
        memory_of_energy = copy.deepcopy(self.memory_of_energy)
        memory_of_restrain_contribution = copy.deepcopy(
            self.memory_of_restrain_contribution
        )
        self.memory_of_energy = []
        self.memory_of_restrain_contribution = []

        if show_plot:
            plotting(
                memory_of_energy,
                memory_of_restrain_contribution,
                "energy [kT]",
                "energy",
                "restrain energy [kT]",
                "restrain",
                "Energy/Ensemble stddev vs minimization step",
            )

        return (np.asarray([f.x.reshape(-1, 3)]) * unit.angstrom, memory_of_energy)

    def calculate_force(
        self, x: simtk.unit.quantity.Quantity, lambda_value: float = 0.0
    ):
        """
        Given a coordinate set the forces with respect to the coordinates are calculated.

        Arguments:
            x {simtk.unit.quantity.Quantity} -- coordinates as 3*(nr of atoms) torch.Tensor.

        Keyword Arguments:
            lambda_value {float} -- position in the lambda protocol (default: {0.0}).

        Raises:
            RuntimeError: raised if self.platform is not specified.

        Returns:
            NamedTuple -- DecomposedForce
        """

        assert type(x) == unit.Quantity
        assert float(lambda_value) <= 1.0 and float(lambda_value) >= 0.0
        x = x.value_in_unit(unit.nanometer)
        if not (len(x.shape) == 3 and x.shape[2] == 3 and x.shape[0] == 1):
            raise RuntimeError(f"Shape of coordinates: {x.shape} is wrong. Aborting.")

        coordinates = torch.tensor(
            x, requires_grad=True, device=self.device, dtype=torch.float32
        )

        energy_in_kT, restraint_energy_contribution = self._calculate_energy(
            coordinates, lambda_value, original_neural_network=True
        )

        # derivative of E (kJ_mol) w.r.t. coordinates (in nm)
        derivative = torch.autograd.grad(
            (energy_in_kT * kT_to_kJ_mol).sum(),
            coordinates,
        )[0]

        if self.platform == "cpu":
            F = -np.array(derivative)[0]
        elif self.platform == "cuda":
            F = -np.array(derivative.cpu())[0]
        else:
            raise RuntimeError("Platform needs to be specified. Either CPU or CUDA.")

        return DecomposedForce(
            (F) * (unit.kilojoule_per_mole / unit.nanometer),
            energy_in_kT.item() * kT,
            restraint_energy_contribution.item() * kT,
        )

    def _calculate_energy(
        self,
        coordinates: torch.Tensor,
        lambda_value: float,
        original_neural_network: bool,
        include_restraint_energy_contribution: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helpter function to return energies as tensor.
        Given a coordinate set the energy is calculated.

        Parameters
        ----------
        coordinates : torch.tensor
            coordinates in nanometer without units attached
        lambda_value : float
            between 0.0 and 1.0
        Returns
        -------
        energy_in_kT : torch.tensor
            return the energy with restraints added
        restraint_energy_contribution_in_kT : torch.tensor
            return the energy of the added restraints
        """

        nr_of_mols = len(coordinates)
        logger.debug(f"len(coordinates): {nr_of_mols}")
        batch_species = torch.stack(
            [self.species[0]] * nr_of_mols
        )  # species is a [1][1] tensor, afterwards it's a [1][nr_of_mols]

        if batch_species.size()[:2] != coordinates.size()[:2]:
            raise RuntimeError(
                f"Dimensions of coordinates: {coordinates.size()} and batch_species: {batch_species.size()} are not the same."
            )

        _, energy_in_hartree = self.model(
            (
                batch_species,
                coordinates * nm_to_angstroms,
                lambda_value,
                original_neural_network,
            )
        )

        # convert energy from hartree to kT
        energy_in_kT = energy_in_hartree * hartree_to_kT
        if include_restraint_energy_contribution:
            restraint_energy_contribution_in_kT = self._compute_restraint_bias(
                coordinates, lambda_value=lambda_value
            )
        else:
            restraint_energy_contribution_in_kT = torch.tensor(
                [0.0] * nr_of_mols, device=self.device, dtype=torch.float64
            )

        energy_in_kT += restraint_energy_contribution_in_kT
        return energy_in_kT, restraint_energy_contribution_in_kT

    def _traget_energy_function(self, x, lambda_value: float = 0.0):
        """
        Given a coordinate set (x) the energy is calculated in kJ/mol.

        Parameters
        ----------
        x : array of floats, unit'd (distance unit)
            initial configuration
        lambda_value : float
            between 0.0 and 1.0 - at zero contributions of alchemical atoms are zero

        Returns
        -------
        E : kT
        F : unit'd
        """
        x = np.asarray([x.reshape(-1, 3)]) * unit.angstrom
        force_energy = self.calculate_force(x, lambda_value)
        F_flat = -np.array(
            force_energy.force.value_in_unit(
                unit.kilojoule_per_mole / unit.angstrom
            ).flatten(),
            dtype=np.float64,
        )
        self.memory_of_energy.append(force_energy.energy)
        self.memory_of_restrain_contribution.append(
            force_energy.restraint_energy_contribution
        )
        return (force_energy.energy.value_in_unit(unit.kilojoule_per_mole), F_flat)

    def calculate_energy(
        self,
        coordinate_list: unit.Quantity,
        lambda_value: float = 0.0,
        original_neural_network: bool = True,
        requires_grad_wrt_coordinates: bool = True,
        requires_grad_wrt_parameters: bool = True,
        include_restraint_energy_contribution: bool = True,
    ):
        """
        Given a coordinate set (x) the energy is calculated in kJ/mol.

        Parameters
        ----------
        x : list, [N][K][3] unit'd (distance unit)
            initial configuration
        lambda_value : float
            between 0.0 and 1.0 - at zero contributions of alchemical atoms are zero

        Returns
        -------
        NamedTuple
        """

        assert type(coordinate_list) == unit.Quantity
        assert 0.0 <= float(lambda_value) <= 1.0
        logger.debug(f"Including restraints: {include_restraint_energy_contribution}")

        logger.debug(f"Batch-size: {len(coordinate_list)}")

        coordinates = torch.tensor(
            coordinate_list.value_in_unit(unit.nanometer),
            requires_grad=requires_grad_wrt_coordinates,
            device=self.device,
            dtype=torch.float32,
        )
        logger.debug(f"coordinates tensor: {coordinates.size()}")

        energy_in_kT, restraint_energy_contribution_in_kT = self._calculate_energy(
            coordinates,
            lambda_value,
            original_neural_network,
            include_restraint_energy_contribution,
        )

        energy = np.array([e.item() for e in energy_in_kT]) * kT

        restraint_energy_contribution = (
            np.array([e.item() for e in restraint_energy_contribution_in_kT]) * kT
        )
        if requires_grad_wrt_parameters:
            return DecomposedEnergy(energy, restraint_energy_contribution, energy_in_kT)
        else:
            return DecomposedEnergy(
                energy, restraint_energy_contribution, energy_in_kT.detach()
            )


class CompartimentedAlchemicalANI2x(AlchemicalANI_Mixin, ANI2x):

    name = "CompartimentedAlchemicalANI2x"

    def __init__(
        self,
        alchemical_atoms: list,
        periodic_table_index: bool = False,
        split_at: int = 6,
        training: bool = False,
    ):
        """Scale the indirect contributions of alchemical atoms to the energy sum by
        linearly interpolating, for other atom i, between the energy E_i^0 it would compute
        in the _complete absence_ of the alchemical atoms, and the energy E_i^1 it would compute
        in the _presence_ of the alchemical atoms.
        (Also scale direct contributions, as in DirectAlchemicalANI)

        Parameters
        ----------
        alchemical_atoms : list
        """

        assert len(alchemical_atoms) == 2
        super().__init__(periodic_table_index)
        self.alchemical_atoms: list = alchemical_atoms
        self.neural_networks = None
        assert self.neural_networks == None
        self.precalculation: dict = {}
        self.split_at: int = split_at
        self.training: bool = training
        self.ANIFirstPart, _ = self.break_into_two_stages(
            self.optimized_neural_network, split_at=self.split_at
        )  # only keep the first part since this is always the same

    def _forward(self, nn, mod_species, mod_coordinates):
        _, ANILastPart = self.break_into_two_stages(
            nn, split_at=self.split_at
        )  # only keep

        species_coordinates = (mod_species, mod_coordinates)

        coordinate_hash = hash(tuple(mod_coordinates[0].flatten().tolist()))

        if coordinate_hash in self.precalculation:
            species, y = self.precalculation[coordinate_hash]
        else:
            species, y = self.ANIFirstPart.forward(species_coordinates)
            self.precalculation[coordinate_hash] = (species, y)

        if self.training:
            # detach so we don't compute expensive gradients w.r.t. y
            species_y = SpeciesEnergies(species, y.detach())
        else:
            species_y = SpeciesEnergies(species, y)

        return ANILastPart.forward(species_y)

    def break_into_two_stages(
        self, model: ANIModel, split_at: int
    ) -> Tuple[Precomputation, LastLayersComputation]:
        """ANIModel.forward(...) is pretty expensive, and in some cases we might want
        to do a computation where the first stage of the calculation is pretty expensive
        and the subsequent stages are less expensive.

        Break ANIModel up into two stages f and g so that
        ANIModel.forward(x) == g.forward(f.forward(x))

        This is beneficial if we only ever need to recompute and adjust g, not f
        """

        if split_at == 6:
            logger.debug("Split at layer 6")
            index_of_last_layers = -1
            nr_of_included_layers = 6
        elif split_at == 4:
            logger.debug("Split at layer 4")
            index_of_last_layers = -3
            nr_of_included_layers = 4
        else:
            raise RuntimeError("Either split at layer 4 or 6.")

        f = Precomputation(
            (model, self.species_converter, self.aev_computer),
            nr_of_included_layers=nr_of_included_layers,
        )
        g = LastLayersComputation(
            (model, self.energy_shifter),
            index_of_last_layers=index_of_last_layers,
            name=self.name,
        )

        return f, g


class CompartimentedAlchemicalANI1ccx(AlchemicalANI_Mixin, ANI1ccx):

    name = "CompartimentedAlchemicalANI1ccx"

    def __init__(
        self,
        alchemical_atoms: list,
        periodic_table_index: bool = False,
        split_at: int = 6,
        training: bool = False,
    ):
        """Scale the indirect contributions of alchemical atoms to the energy sum by
        linearly interpolating, for other atom i, between the energy E_i^0 it would compute
        in the _complete absence_ of the alchemical atoms, and the energy E_i^1 it would compute
        in the _presence_ of the alchemical atoms.
        (Also scale direct contributions, as in DirectAlchemicalANI)

        Parameters
        ----------
        alchemical_atoms : list
        """

        assert len(alchemical_atoms) == 2
        super().__init__(periodic_table_index)
        self.alchemical_atoms: list = alchemical_atoms
        self.neural_networks = None
        assert self.neural_networks == None
        self.precalculation: dict = {}
        self.split_at: int = split_at
        self.training: bool = training
        self.ANIFirstPart, _ = self.break_into_two_stages(
            self.optimized_neural_network, split_at=self.split_at
        )  # only keep the first part since this is always the same

    def _forward(self, nn, mod_species, mod_coordinates):
        _, ANILastPart = self.break_into_two_stages(
            nn, split_at=self.split_at
        )  # only keep

        species_coordinates = (mod_species, mod_coordinates)

        coordinate_hash = hash(tuple(mod_coordinates[0].flatten().tolist()))

        if coordinate_hash in self.precalculation:
            species, y = self.precalculation[coordinate_hash]
        else:
            species, y = self.ANIFirstPart.forward(species_coordinates)
            self.precalculation[coordinate_hash] = (species, y)

        if self.training:
            # detach so we don't compute expensive gradients w.r.t. y
            species_y = SpeciesEnergies(species, y.detach())
        else:
            species_y = SpeciesEnergies(species, y)

        return ANILastPart.forward(species_y)

    def break_into_two_stages(
        self, model: ANIModel, split_at: int
    ) -> Tuple[Precomputation, LastLayersComputation]:
        """ANIModel.forward(...) is pretty expensive, and in some cases we might want
        to do a computation where the first stage of the calculation is pretty expensive
        and the subsequent stages are less expensive.

        Break ANIModel up into two stages f and g so that
        ANIModel.forward(x) == g.forward(f.forward(x))

        This is beneficial if we only ever need to recompute and adjust g, not f
        """

        if split_at == 6:
            logger.debug("Split at layer 6")
            index_of_last_layers = -1
            nr_of_included_layers = 6
        elif split_at == 4:
            logger.debug("Split at layer 4")
            index_of_last_layers = -3
            nr_of_included_layers = 4
        else:
            raise RuntimeError("Either split at layer 4 or 6.")

        f = Precomputation(
            (model, self.species_converter, self.aev_computer),
            nr_of_included_layers=nr_of_included_layers,
        )
        g = LastLayersComputation(
            (model, self.energy_shifter),
            index_of_last_layers=index_of_last_layers,
            name=self.name,
        )

        return f, g
