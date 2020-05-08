import copy
import logging
import os
import random
from collections import namedtuple
from functools import partial
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
from enum import Enum

from .constants import (eV_to_kJ_mol, device,
                        hartree_to_kJ_mol, kT, nm_to_angstroms, platform,
                        pressure, temperature, hartree_to_kT, kT_to_kJ_mol)
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
    energy : unit.Quantity
    restraint_bias: unit.Quantity
    stddev: unit.Quantity
    ensemble_bias: unit.Quantity

class DecomposedEnergy(NamedTuple):
    """Returned by _calculate_energy()

    energy: unit'd
    restraint_bias: unit'd (energy contribution of all restraints)
    stddev: unit'd
    ensemble_bias: unit'd
    energy_tensor: in kT
    """

    energy: unit.Quantity
    restraint_bias: unit.Quantity
    stddev: unit.Quantity
    ensemble_bias: unit.Quantity
    energy_tensor: Tensor

class ANI1_force_and_energy(object):

    def __init__(self,
                 model: torchani.models.ANI1ccx,
                 atoms: str,
                 mol: Atoms = None,
                 adventure_mode: bool = True,
                 per_atom_thresh: unit.Quantity = 0.5 * unit.kilojoule_per_mole,
                 ):
        """
        Performs energy and force calculations.

        Parameters
        ----------
        model: torchani.models
        atoms: str
            a string of atoms in the indexed order
        mol (optional): ase.Atoms
            a ASE Atoms object with the atoms
        adventure_mode :bool
            stddev threshold for energy prediction
        per_atom_thresh: unit'd
        """
        self.device = device
        self.model = model
        self.atoms = atoms
        self.ase_mol = mol
        self.species = self.model.species_to_tensor(atoms).to(device).unsqueeze(0)
        self.platform = platform
        self.list_of_lambda_restraints:list = []
        self.per_atom_thresh = per_atom_thresh.value_in_unit(unit.kilojoule_per_mole)
        self.adventure_mode = adventure_mode
        self.per_mol_tresh = float(self.per_atom_thresh * len(self.atoms))

        assert(type(per_atom_thresh) == unit.Quantity)

        # TODO: check availablity of platform

    def add_restraint_to_lambda_protocol(self, restraint):
        """
        Add a single restraint to the lambda protocol.

        Arguments:
            restraint {neutromeratio.restraint.Restraint} -- Either Harmonic or FlatBottomRestraint
        """
        assert(isinstance(restraint, BaseDistanceRestraint))
        self.list_of_lambda_restraints.append(restraint)

    def reset_lambda_restraints(self):
        """
        Resets the restraints for the lambda protocol
        """
        self.list_of_lambda_restraints = []

    def _compute_restraint_bias(self, x, lambda_value):
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
        lambda_restraint_bias_in_kT = torch.tensor(0.0,
                                                       device=self.device, dtype=torch.float64)

        for restraint in self.list_of_lambda_restraints:
            restraint_bias = restraint.restraint(x * nm_to_angstroms)
            if restraint.active_at == 1:
                restraint_bias *= lambda_value
            elif restraint.active_at == 0:
                restraint_bias *= (1 - lambda_value)
            elif restraint.active_at == -1:  # always on
                pass
            else:
                raise RuntimeError('Something went wrong with restraints.')
            lambda_restraint_bias_in_kT += (restraint_bias * unit.kilojoule_per_mole)/kT


        return lambda_restraint_bias_in_kT

    def compute_restraint_bias_on_snapshots(self, snapshots, lambda_value=0.0) -> float:
        """
        Calculates the energy of all restraint_bias activate at lambda_value on a given snapshot.
        Returns
        -------
        reduced_restraint_bias : float
            the restraint_bias in kT
        """
        restraint_bias_in_kJ_mol = list(
            map(partial(self._compute_restraint_bias, lambda_value=lambda_value), snapshots))
        unitted_restraint_bias = np.array(list(map(float, restraint_bias_in_kJ_mol))) * unit.kilojoules_per_mole
        reduced_restraint_bias = unitted_restraint_bias / kT

        return reduced_restraint_bias

    def get_thermo_correction(self, coords: simtk.unit.quantity.Quantity) -> unit.quantity.Quantity:
        
        """        
        Returns the thermochemistry correction. This calls: https://wiki.fysik.dtu.dk/ase/ase/thermochemistry/thermochemistry.html
        and uses the Ideal gas rigid rotor harmonic oscillator approximation to calculate the Gibbs free energy correction that 
        needs to be added to the single point energy to obtain the Gibb's free energy

        Raises:
            verror: if imaginary frequencies are detected a ValueError is raised

        Returns:
            float -- temperature correct [kT] 
        """

        ase_mol = copy.deepcopy(self.ase_mol)
        for atom, c in zip(ase_mol, coords):
            atom.x = c[0].value_in_unit(unit.angstrom)
            atom.y = c[1].value_in_unit(unit.angstrom)
            atom.z = c[2].value_in_unit(unit.angstrom)

        calculator = self.model.ase()
        ase_mol.set_calculator(calculator)

        vib = Vibrations(ase_mol, name=f"/tmp/vib{random.randint(1,10000000)}")
        vib.run()
        vib_energies = vib.get_energies()
        thermo = IdealGasThermo(vib_energies=vib_energies,
                                atoms=ase_mol,
                                geometry='nonlinear',
                                symmetrynumber=1, spin=0)

        try:
            G = thermo.get_gibbs_energy(temperature=temperature.value_in_unit(
                unit.kelvin), pressure=pressure.value_in_unit(unit.pascal))
        except ValueError as verror:
            print(verror)
            vib.clean()
            raise verror
        # removes the vib tmp files
        vib.clean()
        return ((G * eV_to_kJ_mol) * unit.kilojoule_per_mole)  # eV * conversion_factor(eV to kJ/mol)

    def minimize(self, coords: simtk.unit.quantity.Quantity, maxiter: int = 1000,
                lambda_value: float = 0.0, show_plot: bool = False)->Tuple[simtk.unit.quantity.Quantity, list]:
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
        assert(type(coords) == unit.Quantity)

        def plotting(y1, y2, y1_axis_label, y1_label, y2_axis_label, y2_label, title):
            fig, ax1 = plt.subplots()
            plt.title(title)

            color = 'tab:red'
            ax1.set_xlabel('timestep (10 fs)')
            ax1.set_ylabel(y1_axis_label, color=color)
            plt.plot([e / kT for e in y1], label=y1_label, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            ax2.set_ylabel(y2_axis_label, color=color)  # we already handled the x-label with ax1
            plt.plot([e / kT for e in y2], label=y2_label, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_xlabel('timestep')
            plt.legend()
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()
            plt.close()

        x = coords.value_in_unit(unit.angstrom)
        self.memory_of_energy:list = []
        self.memory_of_stddev:list = []
        self.memory_of_ensemble_bias:list = []
        self.memory_of_conformation_bias: list = []
        
        print("Begin minimizing...")
        f = optimize.minimize(self._traget_energy_function, x, method='BFGS',
                              jac=True, args=(lambda_value),
                              options={'maxiter': maxiter, 'disp': True})

        logger.critical(f"Minimization status: {f.success}")
        memory_of_energy = copy.deepcopy(self.memory_of_energy)
        memory_of_stddev = copy.deepcopy(self.memory_of_stddev)
        memory_of_ensemble_bias = copy.deepcopy(self.memory_of_ensemble_bias)
        memory_of_conformation_bias = copy.deepcopy(self.memory_of_conformation_bias)
        self.memory_of_energy = []
        self.memory_of_stddev = []
        self.memory_of_ensemble_bias = []
        self.memory_of_conformation_bias = []

        if show_plot:
            # plot 1
            plotting(memory_of_energy, memory_of_stddev,
                     'energy [kT]', 'energy', 'stddev', 'ensemble stddev [kT]', 'Energy/Ensemble stddev vs minimization step')
            # plot 2
            plotting(memory_of_ensemble_bias, memory_of_stddev,
                     'penelty [kT]', 'ensemble_bias', 'stddev', 'ensemble stddev [kT]', 'Ensemble bias/Ensemble stddev vs minimization step')
            # plot 3
            plotting(memory_of_energy, memory_of_ensemble_bias,
                     'energy [kT]', 'energy', 'ensemble bias [kT]', 'ensemble bias', 'Ensemble bias/Energy vs minimization step')
            # plot 4
            plotting(memory_of_energy, memory_of_conformation_bias,
                     'energy [kT]', 'energy', 'restraint bias [kT]', 'restraint bias', 'Restraint bias/Energy vs minimization step')

        return (f.x.reshape(-1, 3) * unit.angstrom, memory_of_energy)

    def calculate_force(self, x: simtk.unit.quantity.Quantity, lambda_value: float = 0.0):
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
        
        
        assert(type(x) == unit.Quantity)
        assert(float(lambda_value) <= 1.0 and float(lambda_value) >= 0.0)

        coordinates = torch.tensor([x.value_in_unit(unit.nanometer)],
                                   requires_grad=True, device=self.device, dtype=torch.float32)

        energy_in_kT, restraint_bias_in_kT, stddev_in_kT, ensemble_bias_kT = self._calculate_energy(
            coordinates, lambda_value)

        # derivative of E (kJ_mol) w.r.t. coordinates (in nm)
        derivative = torch.autograd.grad(((energy_in_kT  *kT).value_in_unit(unit.kilojoule_per_mole)).sum(), coordinates)[0]

        if self.platform == 'cpu':
            F = - np.array(derivative)[0]
        elif self.platform == 'cuda':
            F = - np.array(derivative.cpu())[0]
        else:
            raise RuntimeError('Platform needs to be specified. Either CPU or CUDA.')

        return DecomposedForce((F) * (unit.kilojoule_per_mole / unit.nanometer),
                energy_in_kT.item() *kT,
                restraint_bias_in_kT.item() *kT,
                stddev_in_kT.item() *kT,
                ensemble_bias_kT.item() *kT) 


    def _calculate_energy(self, coordinates: torch.Tensor, lambda_value: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        stddev_in_hartree = torch.tensor(0.0,
                                         device=self.device, dtype=torch.float64)

        restraint_bias_in_kT = torch.tensor(0.0,
                                                device=self.device, dtype=torch.float64)

        ensemble_bias_in_kT = torch.tensor(0.0,
                                               device=self.device, dtype=torch.float64)

        assert(0.0 <= float(lambda_value) <= 1.0)

        _, energy_in_hartree, stddev_in_hartree = self.model(
            (self.species, coordinates * nm_to_angstroms, lambda_value))

        # convert energy from hartree to kT
        energy_in_kT = energy_in_hartree * hartree_to_kT
        stddev_in_kT = stddev_in_hartree * hartree_to_kT

        restraint_bias_in_kT = self._compute_restraint_bias(
            coordinates, lambda_value=lambda_value)

        energy_in_kT += restraint_bias_in_kT

        if self.adventure_mode == False:
            if stddev_in_kT > self.per_mol_tresh:
                #logger.info(f"Per atom tresh: {self.per_atom_thresh}")
                #logger.info(f"Nr of atoms: {species.size()[1]}")
                #logger.warning(f"Stddev: {stddev_in_kJ_mol} kJ/mol")
                #logger.warning(f"Energy: {energy_in_kJ_mol} kJ/mol")
                ensemble_bias_in_kT = self._linear_ensemble_bias(stddev_in_kT)

            energy_in_kT += ensemble_bias_in_kT

        return energy_in_kT, restraint_bias_in_kT, stddev_in_kT, ensemble_bias_in_kT

    def _quadratic_ensemble_bias(self, stddev):
        ensemble_bias_in_kT = torch.tensor(0.1 * ((stddev.item() - self.per_mol_tresh)**2),
                                               device=self.device, dtype=torch.float64, requires_grad=True)
        logger.warning(f"Applying ensemble_bias: {ensemble_bias_in_kT.item()} kT")
        return ensemble_bias_in_kT

    def _linear_ensemble_bias(self, stddev):
        ensemble_bias_in_kT = torch.tensor(abs(stddev.item() - self.per_mol_tresh),
                                               device=self.device, dtype=torch.float64, requires_grad=True)
        logger.warning(f"Applying ensemble_bias: {ensemble_bias_in_kT.item()} kT")
        return ensemble_bias_in_kT

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
        x = x.reshape(-1, 3) * unit.angstrom
        force_energy = self.calculate_force(x, lambda_value)
        F_flat = -np.array(force_energy.force.value_in_unit(unit.kilojoule_per_mole/unit.angstrom).flatten(), dtype=np.float64)
        self.memory_of_energy.append(force_energy.energy)
        self.memory_of_stddev.append(force_energy.stddev)
        self.memory_of_conformation_bias.append(force_energy.restraint_bias)
        self.memory_of_ensemble_bias.append(force_energy.ensemble_bias)
        return (force_energy.energy.value_in_unit(unit.kilojoule_per_mole), F_flat)

    def calculate_energy(self, x: simtk.unit.quantity.Quantity, lambda_value: float = 0.0):
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
        NamedTuple
        """

        assert(type(x) == unit.Quantity)

        coordinates = torch.tensor([x.value_in_unit(unit.nanometer)],
                                   requires_grad=True, device=self.device, dtype=torch.float32)

        energy_in_kT, restraint_bias_in_kT, stddev_in_kT, ensemble_bias_in_kT = self._calculate_energy(
            coordinates, lambda_value)

        energy = (energy_in_kT.item() *kT)
        restraint_bias = (restraint_bias_in_kT.item()  *kT)
        stddev = (stddev_in_kT.item()  *kT)
        ensemble_bias = (ensemble_bias_in_kT.item()  *kT)

        return DecomposedEnergy(energy, restraint_bias, stddev, ensemble_bias, energy_in_kT)


class AlchemicalANI(torchani.models.ANI1ccx):

    def __init__(self, alchemical_atoms=[]):
        """Scale the contributions of alchemical atoms to the energy."""
        super().__init__()
        self.alchemical_atoms = alchemical_atoms

    def forward(self, species_coordinates, lam=1.0):
        raise (NotImplementedError)


class PureANI1ccx(torchani.models.ANI1ccx):
    def __init__(self):
        """
        Pure ANI1ccx model with ensemble stddev
        """
        super().__init__()
        self.neural_networks = load_model_ensemble(self.species,
                                                   self.ensemble_prefix,
                                                   self.ensemble_size
                                                   )
        self.device = device

    def forward(self, species_coordinates):

        # species, AEVs of fully interacting system
        try:
            species, coordinates, _ = species_coordinates
        except ValueError:
            species, coordinates = species_coordinates

        aevs = (species, coordinates)
        species, aevs = self.aev_computer(aevs)

        # neural net output given these AEVs
        state = self.neural_networks((species, aevs))
        _, E = self.energy_shifter((species, state.energies))

        return SpeciesEnergies(species, E, state.stddev)


class PureANI1x(torchani.models.ANI1x):
    def __init__(self):
        """
        Pure ANI1x model with ensemble stddev
        """
        super().__init__()
        self.neural_networks = load_model_ensemble(self.species,
                                                   self.ensemble_prefix,
                                                   self.ensemble_size
                                                   )
        self.device = device

    def forward(self, species_coordinates):

        # species, AEVs of fully interacting system
        try:
            species, coordinates, _ = species_coordinates
        except ValueError:
            species, coordinates = species_coordinates
        aevs = (species, coordinates)
        species, aevs = self.aev_computer(aevs)

        # neural net output given these AEVs
        state = self.neural_networks((species, aevs))
        _, E = self.energy_shifter((species, state.energies))

        return SpeciesEnergies(species, E, state.stddev)




class Ensemble(torch.nn.ModuleList):
    """Compute the average output of an ensemble of modules."""

    def __init__(self, modules):
        super().__init__(modules)
        self.size = len(modules)

    def forward(self, species_input: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> (torch.Tensor, torch.Tensor):
        """
        Returns the averager and mean of the NN ensemble energy prediction
        Returns
        -------
        energy_mean : torch.Tensor in Hartree
        stddev : torch.Tensor in Hartree
        """

        outputs_tensor = torch.cat([x(species_input)[1].double() for x in self])
        stddev = torch.std(outputs_tensor, unbiased=False)  # to match np.std default ddof=0
        energy_mean = torch.mean(outputs_tensor)
        species, _ = species_input
        return SpeciesEnergies(species, energy_mean, stddev)


def load_model_ensemble(species, prefix, count):
    """Returns an instance of :class:`torchani.Ensemble` loaded from
    NeuroChem's network directories beginning with the given prefix.
    Arguments:
        species (:class:`collections.abc.Sequence`): Sequence of strings for
            chemical symbols of each supported atom type in correct order.
        prefix (str): Prefix of paths of directory that networks configurations
            are stored.
        count (int): Number of models in the ensemble.
    """
    models = []
    for i in range(count):
        network_dir = os.path.join('{}{}'.format(prefix, i), 'networks')
        models.append(torchani.neurochem.load_model(species, network_dir))
    return Ensemble(models)


class LinearAlchemicalANI(AlchemicalANI):

    def __init__(self, alchemical_atoms: list):
        """Scale the indirect contributions of alchemical atoms to the energy sum by
        linearly interpolating, for other atom i, between the energy E_i^0 it would compute
        in the _complete absence_ of the alchemical atoms, and the energy E_i^1 it would compute
        in the _presence_ of the alchemical atoms.
        (Also scale direct contributions, as in DirectAlchemicalANI)
        """
        super().__init__(alchemical_atoms)
        self.neural_networks = load_model_ensemble(self.species,
                                                   self.ensemble_prefix,
                                                   self.ensemble_size
                                                   )
        self.device = device

    def forward(self, species_coordinates):

        assert(len(self.alchemical_atoms) == 1)
        alchemical_atom = self.alchemical_atoms[0]

        # LAMBDA = 1: fully interacting
        # species, AEVs of fully interacting system
        species, coordinates, lam = species_coordinates
        aevs = (species, coordinates)
        species, aevs = self.aev_computer(aevs)

        # neural net output given these AEVs
        state_1 = self.neural_networks((species, aevs))
        _, E_1 = self.energy_shifter((species, state_1.energies))

        # LAMBDA == 1: fully interacting
        if float(lam) == 1.0:
            E = E_1
            stddev = state_1.stddev
        else:
            # LAMBDA == 0: fully removed
            # species, AEVs of all other atoms, in absence of alchemical atoms
            mod_species = torch.cat((species[:, :alchemical_atom],  species[:, alchemical_atom+1:]), dim=1)
            mod_coordinates = torch.cat((coordinates[:, :alchemical_atom],  coordinates[:, alchemical_atom+1:]), dim=1)
            _, mod_aevs = self.aev_computer((mod_species, mod_coordinates))
            # neural net output given these modified AEVs
            state_0 = self.neural_networks((mod_species, mod_aevs))
            _, E_0 = self.energy_shifter((species, state_0.energies))
            E = (lam * E_1) + ((1 - lam) * E_0)
            stddev = (lam * state_1.stddev) + ((1-lam) * state_0.stddev)

        return species, E, stddev


class LinearAlchemicalSingleTopologyANI(LinearAlchemicalANI):

    def __init__(self, alchemical_atoms: list):
        """Scale the indirect contributions of alchemical atoms to the energy sum by
        linearly interpolating, for other atom i, between the energy E_i^0 it would compute
        in the _complete absence_ of the alchemical atoms, and the energy E_i^1 it would compute
        in the _presence_ of the alchemical atoms.
        (Also scale direct contributions, as in DirectAlchemicalANI)

        Parameters
        ----------
        alchemical_atoms : list
        adventure_mode : bool
            “Fortune and glory, kid. Fortune and glory.” - Indiana Jones
        """

        assert(len(alchemical_atoms) == 2)

        super().__init__(alchemical_atoms)

    def forward(self, species_coordinates):
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

        # species, AEVs of fully interacting system
        species, coordinates, lam = species_coordinates
        # NOTE: I am not happy about this - the order at which
        # the dummy atoms are set in alchemical_atoms determines
        # what is real and what is dummy at lambda 1 - that seems awefully error prone
        dummy_atom_0 = self.alchemical_atoms[0]
        dummy_atom_1 = self.alchemical_atoms[1]

        # neural net output given these AEVs
        mod_species_0 = torch.cat((species[:, :dummy_atom_0],  species[:, dummy_atom_0+1:]), dim=1)
        mod_coordinates_0 = torch.cat((coordinates[:, :dummy_atom_0],  coordinates[:, dummy_atom_0+1:]), dim=1)
        _, mod_aevs_0 = self.aev_computer((mod_species_0, mod_coordinates_0))
        # neural net output given these modified AEVs
        state_0 = self.neural_networks((mod_species_0, mod_aevs_0))
        _, E_0 = self.energy_shifter((mod_species_0, state_0.energies))

        # neural net output given these AEVs
        mod_species_1 = torch.cat((species[:, :dummy_atom_1],  species[:, dummy_atom_1+1:]), dim=1)
        mod_coordinates_1 = torch.cat((coordinates[:, :dummy_atom_1],  coordinates[:, dummy_atom_1+1:]), dim=1)
        _, mod_aevs_1 = self.aev_computer((mod_species_1, mod_coordinates_1))
        # neural net output given these modified AEVs
        state_1 = self.neural_networks((mod_species_1, mod_aevs_1))
        _, E_1 = self.energy_shifter((mod_species_1, state_1.energies))

        E = (lam * E_1) + ((1 - lam) * E_0)
        stddev = (lam * state_1.stddev) + ((1-lam) * state_0.stddev)
        return species, E, stddev
