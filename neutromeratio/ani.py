import torch
import copy
import logging
import os
import random
from collections import namedtuple
from functools import partial
from typing import NamedTuple, Optional, Tuple
import copy

import matplotlib.pyplot as plt
import numpy as np
import simtk
import torchani
from ase import Atoms
from ase.thermochemistry import IdealGasThermo
from ase.vibrations import Vibrations
from simtk import unit
from torch import Tensor
from enum import Enum

from .constants import (
    eV_to_kJ_mol,
    device,
    hartree_to_kJ_mol,
    kT,
    nm_to_angstroms,
    platform,
    pressure,
    temperature,
    hartree_to_kT,
    kT_to_kJ_mol,
)
from .restraints import BaseDistanceRestraint

logger = logging.getLogger(__name__)


class SpeciesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor
    stddev: Tensor


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

    def load_nn_parameters(
        self, parameter_path: str, extract_from_checkpoint: bool = False
    ):
        if os.path.isfile(parameter_path):
            parameters = torch.load(parameter_path)
            if extract_from_checkpoint:
                self.tweaked_neural_network.load_state_dict(parameters["nn"])
            else:
                self.tweaked_neural_network.load_state_dict(parameters)
        else:
            logger.info(f"Parameter file {parameter_path} does not exist.")

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
            nn = self.tweaked_neural_network
            logger.debug("Using possibly tweaked neural network parameters.")

        species_coordinates = (species, coordinates)
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=None, pbc=None)
        species_energies = nn(species_aevs)
        return self.energy_shifter(species_energies)


class ANI1x(ANI):

    tweaked_neural_network = None
    original_neural_network = None
    name = "ANI1x"

    def __init__(self, periodic_table_index: bool = False):
        info_file = "ani-1x_8x.info"
        super().__init__(info_file, periodic_table_index)
        if ANI1x.tweaked_neural_network == None:
            ANI1x.tweaked_neural_network = copy.deepcopy(self.neural_networks)
        if ANI1x.original_neural_network == None:
            ANI1x.original_neural_network = copy.deepcopy(self.neural_networks)


class ANI1ccx(ANI):

    tweaked_neural_network = None
    original_neural_network = None

    def __init__(self, periodic_table_index: bool = False):
        info_file = "ani-1ccx_8x.info"
        super().__init__(info_file, periodic_table_index)
        if ANI1ccx.tweaked_neural_network == None:
            ANI1ccx.tweaked_neural_network = copy.deepcopy(self.neural_networks)
        if ANI1ccx.original_neural_network == None:
            ANI1ccx.original_neural_network = copy.deepcopy(self.neural_networks)


class ANI2x(ANI):
    tweaked_neural_network = None
    original_neural_network = None
    name = "ANI2x"

    def __init__(self, periodic_table_index: bool = False):
        info_file = "ani-2x_8x.info"
        super().__init__(info_file, periodic_table_index)
        if ANI2x.tweaked_neural_network == None:
            ANI2x.tweaked_neural_network = copy.deepcopy(self.neural_networks)
        if ANI2x.original_neural_network == None:
            ANI2x.original_neural_network = copy.deepcopy(self.neural_networks)


class AlchemicalANI1ccx(ANI1ccx):

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

    def _reset_parameters(self):
        self.tweaked_neural_network = copy.deepcopy(self.neural_networks)

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
        species_coordinates = (species, coordinates)

        if original_parameters:
            logger.debug("Using original neural network parameters.")
            nn = self.original_neural_network
        else:
            nn = self.tweaked_neural_network
            logger.debug("Using possibly tweaked neural network parameters.")

        # setting dummy atoms
        dummy_atom_0 = self.alchemical_atoms[0]
        dummy_atom_1 = self.alchemical_atoms[1]

        # neural net output given these AEVs
        mod_species_0 = torch.cat(
            (species[:, :dummy_atom_0], species[:, dummy_atom_0 + 1 :]), dim=1
        )
        mod_coordinates_0 = torch.cat(
            (coordinates[:, :dummy_atom_0], coordinates[:, dummy_atom_0 + 1 :]), dim=1
        )
        _, mod_aevs_0 = self.aev_computer((mod_species_0, mod_coordinates_0))

        # neural net output given these modified AEVs
        state_0 = nn((mod_species_0, mod_aevs_0))
        _, E_0 = self.energy_shifter((mod_species_0, state_0.energies))

        # neural net output given these AEVs
        mod_species_1 = torch.cat(
            (species[:, :dummy_atom_1], species[:, dummy_atom_1 + 1 :]), dim=1
        )
        mod_coordinates_1 = torch.cat(
            (coordinates[:, :dummy_atom_1], coordinates[:, dummy_atom_1 + 1 :]), dim=1
        )
        _, mod_aevs_1 = self.aev_computer((mod_species_1, mod_coordinates_1))

        # neural net output given these modified AEVs
        state_1 = nn((mod_species_1, mod_aevs_1))
        _, E_1 = self.energy_shifter((mod_species_1, state_1.energies))

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

        E = (lam * E_1) + ((1 - lam) * E_0)
        return species, E


class AlchemicalANI1x(ANI1x):

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

    def _reset_parameters(self):
        self.tweaked_neural_network = copy.deepcopy(self.neural_networks)

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
        species_coordinates = (species, coordinates)

        if original_parameters:
            logger.debug("Using original neural network parameters.")
            nn = self.original_neural_network
        else:
            nn = self.tweaked_neural_network
            logger.debug("Using possibly tweaked neural network parameters.")

        # setting dummy atoms
        dummy_atom_0 = self.alchemical_atoms[0]
        dummy_atom_1 = self.alchemical_atoms[1]
        (species, coordinates) = species_coordinates

        # neural net output given these AEVs
        mod_species_0 = torch.cat(
            (species[:, :dummy_atom_0], species[:, dummy_atom_0 + 1 :]), dim=1
        )
        mod_coordinates_0 = torch.cat(
            (coordinates[:, :dummy_atom_0], coordinates[:, dummy_atom_0 + 1 :]), dim=1
        )
        _, mod_aevs_0 = self.aev_computer((mod_species_0, mod_coordinates_0))

        # neural net output given these modified AEVs
        state_0 = nn((mod_species_0, mod_aevs_0))
        _, E_0 = self.energy_shifter((mod_species_0, state_0.energies))

        # neural net output given these AEVs
        mod_species_1 = torch.cat(
            (species[:, :dummy_atom_1], species[:, dummy_atom_1 + 1 :]), dim=1
        )
        mod_coordinates_1 = torch.cat(
            (coordinates[:, :dummy_atom_1], coordinates[:, dummy_atom_1 + 1 :]), dim=1
        )
        _, mod_aevs_1 = self.aev_computer((mod_species_1, mod_coordinates_1))

        # neural net output given these modified AEVs
        state_1 = nn((mod_species_1, mod_aevs_1))
        _, E_1 = self.energy_shifter((mod_species_1, state_1.energies))

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

        E = (lam * E_1) + ((1 - lam) * E_0)
        return species, E


class AlchemicalANI2x(ANI2x):

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
        self.alchemical_atoms = alchemical_atoms
        self.neural_networks = None
        assert self.neural_networks == None

    def _reset_parameters(self):
        self.tweaked_neural_network = copy.deepcopy(self.neural_networks)

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
        assert len(species_coordinates_lamb) == 4
        species, coordinates, lam, original_parameters = species_coordinates_lamb
        species_coordinates = (species, coordinates)

        if original_parameters:
            logger.debug("Using original neural network parameters.")
            nn = self.original_neural_network
        else:
            nn = self.tweaked_neural_network
            logger.debug("Using possibly tweaked neural network parameters.")

        # setting dummy atoms
        dummy_atom_0 = self.alchemical_atoms[0]
        dummy_atom_1 = self.alchemical_atoms[1]
        (species, coordinates) = species_coordinates

        # neural net output given these AEVs
        mod_species_0 = torch.cat(
            (species[:, :dummy_atom_0], species[:, dummy_atom_0 + 1 :]), dim=1
        )
        mod_coordinates_0 = torch.cat(
            (coordinates[:, :dummy_atom_0], coordinates[:, dummy_atom_0 + 1 :]), dim=1
        )
        _, mod_aevs_0 = self.aev_computer((mod_species_0, mod_coordinates_0))

        # neural net output given these modified AEVs
        state_0 = nn((mod_species_0, mod_aevs_0))
        _, E_0 = self.energy_shifter((mod_species_0, state_0.energies))

        # neural net output given these AEVs
        mod_species_1 = torch.cat(
            (species[:, :dummy_atom_1], species[:, dummy_atom_1 + 1 :]), dim=1
        )
        mod_coordinates_1 = torch.cat(
            (coordinates[:, :dummy_atom_1], coordinates[:, dummy_atom_1 + 1 :]), dim=1
        )
        _, mod_aevs_1 = self.aev_computer((mod_species_1, mod_coordinates_1))

        # neural net output given these modified AEVs
        state_1 = nn((mod_species_1, mod_aevs_1))
        _, E_1 = self.energy_shifter((mod_species_1, state_1.energies))

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

        E = (lam * E_1) + ((1 - lam) * E_0)
        return species, E


class ANI1_force_and_energy(object):
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
            x {Tensor} -- coordinates as torch.Tensor
            lambda_value {float} -- lambda value

        Raises:
            RuntimeError: raises RuntimeError if restraint.active_at has numeric value outside [0,1]

        Returns:
            float -- energy [kT]
        """

        # use correct restraint_bias in between the end-points...

        # lambda
        nr_of_mols = len(coordinates)
        restraint_bias_in_kT = torch.tensor(
            [0.0] * nr_of_mols, device=self.device, dtype=torch.float64
        )
        for restraint in self.list_of_lambda_restraints:
            restraint_bias = restraint.restraint(coordinates * nm_to_angstroms)
            if restraint.active_at == 1:
                restraint_bias *= lambda_value
            elif restraint.active_at == 0:
                restraint_bias *= 1 - lambda_value
            elif restraint.active_at == -1:  # always on
                pass
            else:
                raise RuntimeError("Something went wrong with restraints.")
            restraint_bias_in_kT += (restraint_bias * unit.kilojoule_per_mole) / kT
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
            # plot 1
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
            ((energy_in_kT * kT).value_in_unit(unit.kilojoule_per_mole)).sum(),
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        restraint_bias_in_kT : torch.tensor
            return the energy of the added restraints
        stddev_in_kT : torch.tensor
            return the stddev of the energy (without added restraints)
        ensemble_bias_in_kT : torch.tensor
            return the ensemble_bias added to the energy
        """

        nr_of_mols = len(coordinates)
        logger.debug(f"len(coordinates): {nr_of_mols}")
        batch_species = torch.stack([self.species[0]] * nr_of_mols)

        assert 0.0 <= float(lambda_value) <= 1.0
        assert isinstance(original_neural_network, bool)

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

        restraint_energy_contribution = self._compute_restraint_bias(
            coordinates, lambda_value=lambda_value
        )

        energy_in_kT += restraint_energy_contribution

        return energy_in_kT, restraint_energy_contribution

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
        logger.debug(f"Batch-size: {len(coordinate_list)}")

        coordinates = torch.tensor(
            coordinate_list.value_in_unit(unit.nanometer),
            requires_grad=requires_grad_wrt_coordinates,
            device=self.device,
            dtype=torch.float32,
        )

        logger.debug(f"coordinates tensor: {coordinates.size()}")
        energy_in_kT, restraint_energy_contribution = self._calculate_energy(
            coordinates, lambda_value, original_neural_network
        )

        energy = np.array([e.item() for e in energy_in_kT]) * kT
        restraint_energy_contribution = (
            np.array([e.item() for e in restraint_energy_contribution]) * kT
        )

        if requires_grad_wrt_parameters:
            return DecomposedEnergy(energy, restraint_energy_contribution, energy_in_kT)
        else:
            return DecomposedEnergy(
                energy, restraint_energy_contribution, energy_in_kT.detach()
            )
