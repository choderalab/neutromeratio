import os, random
import torchani
import torch
import numpy as np
from .constants import nm_to_angstroms, hartree_to_kJ_mol, device, platform, conversion_factor_eV_to_kJ_mol, temperature, pressure
from simtk import unit
import simtk
from .restraints import Restraint
from ase.optimize import BFGS
from ase import Atoms
import copy
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
import logging

logger = logging.getLogger(__name__)

class ANI1_force_and_energy(object):

    def __init__(self, 
                model:torchani.models.ANI1ccx, 
                atoms:str,
                mol:Atoms,
                use_pure_ani1ccx:bool=False
                ):
        """
        Performs energy and force calculations.
        
        Parameters
        ----------
        model:
        atoms: str
            a string of atoms in the indexed order
        mol: ase.Atoms
            a ASE Atoms object with the atoms
        use_pure_ani1ccx : bool
            a boolian that controlls if a pure ani1ccx model is used
        """


        self.device = device
        self.model = model
        self.atoms = atoms
        self.ase_mol = mol
        self.species = self.model.species_to_tensor(atoms).to(device).unsqueeze(0)
        self.platform = platform
        self.use_pure_ani1ccx = use_pure_ani1ccx       
        self.list_of_restraints = []
        # TODO: check availablity of platform

    def add_restraint(self, restraint:Restraint):
        # add a single restraint
        self.list_of_restraints.append(restraint)

    def reset_restraints(self):
        self.list_of_restraints = []

    def get_thermo_correction(self, coords:simtk.unit.quantity.Quantity) -> unit.quantity.Quantity :
        """
        Returns the thermochemistry correction. This calls: https://wiki.fysik.dtu.dk/ase/ase/thermochemistry/thermochemistry.html
        and uses the Ideal gas rigid rotor harmonic oscillator approximation to calculate the Gibbs free energy correction that 
        needs to be added to the single point energy to obtain the Gibb's free energy

        Parameters
        ----------
        coords:simtk.unit.quantity.Quantity
        Returns
        -------
        gibbs_energy_correction : unit.kilojoule_per_mole
        """

        ase_mol = copy.deepcopy(self.ase_mol)
        for atom, c in zip(ase_mol, coords):
            atom.x = c[0].value_in_unit(unit.angstrom)
            atom.y = c[1].value_in_unit(unit.angstrom)
            atom.z = c[2].value_in_unit(unit.angstrom)

        calculator = self.model.ase(dtype=torch.float64)
        ase_mol.set_calculator(calculator)

        vib = Vibrations(ase_mol, name=f"vib{random.randint(1,10000000)}")
        vib.run()
        vib_energies = vib.get_energies()
        thermo = IdealGasThermo(vib_energies=vib_energies,
                                atoms=ase_mol,
                                geometry='nonlinear',
                                symmetrynumber=1, spin=0)
        
        try:
            G = thermo.get_gibbs_energy(temperature=temperature.value_in_unit(unit.kelvin), pressure=pressure.value_in_unit(unit.pascal))
        except ValueError as verror:
            print(verror)
            raise verror
        #vib.write_jmol()
        vib.clean()
        return (G * conversion_factor_eV_to_kJ_mol) * unit.kilojoule_per_mole # eV * conversion_factor(eV to kJ/mol)


    def minimize(self, coords:simtk.unit.quantity.Quantity, fmax:float=0.001, maxstep:float=0.04):
        """
        Minimizes the molecule.
        Parameters
        ----------
        coords:simtk.unit.quantity.Quantity
        fmax: float
            the final maximum accepted change in energy from t-1 optimization step to t
        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.04 Å). Decrease this value for tricky geometries.
        Returns
        -------
        coords:simtk.unit.quantity.Quantity
        """
        mol = copy.deepcopy(self.ase_mol)
        calculator = self.model.ase(dtype=torch.float64)
        logger.info(f"Fmax set to {fmax}")
        logger.info(f"maxstep set to {maxstep}")

        for atom, c in zip(mol, coords):
            atom.x = c[0].value_in_unit(unit.angstrom)
            atom.y = c[1].value_in_unit(unit.angstrom)
            atom.z = c[2].value_in_unit(unit.angstrom)

        mol.set_calculator(calculator)
        print("Begin minimizing...")
        opt = BFGS(mol, logfile='min-log.txt', maxstep=maxstep)
        opt.run(fmax=fmax)
        return np.array(mol.get_positions()) * unit.angstrom

    def calculate_force(self, x:simtk.unit.quantity.Quantity, lambda_value:float = 0.0) -> simtk.unit.quantity.Quantity:
        """
        Given a coordinate set the forces with respect to the coordinates are calculated.
        
        Parameters
        ----------
        x : array of floats, unit'd (distance unit)
            initial configuration
        lambda_value : float
            between 0.0 and 1.0 - at zero contributions of alchemical atoms are zero
        Returns
        -------
        F : float, unit'd
            
        """
        assert(type(x) == unit.Quantity)
        assert(float(lambda_value) <= 1.0 and float(lambda_value) >= 0.0)

        coordinates = torch.tensor([x.value_in_unit(unit.nanometer)],
                                requires_grad=True, device=self.device, dtype=torch.float32)

        energy_in_kJ_mol, bias_in_kJ_mol = self._calculate_energy(coordinates, lambda_value)

        # derivative of E (in kJ/mol) w.r.t. coordinates (in nm)
        derivative = torch.autograd.grad((energy_in_kJ_mol).sum(), coordinates)[0]

        if self.platform == 'cpu':
            F = - np.array(derivative)[0]
        elif self.platform == 'cuda':
            F = - np.array(derivative.cpu())[0]
        else:
            raise RuntimeError('Platform needs to be specified. Either CPU or CUDA.')

        return F * (unit.kilojoule_per_mole / unit.nanometer), energy_in_kJ_mol.item() * unit.kilojoule_per_mole, bias_in_kJ_mol.item() * unit.kilojoule_per_mole 

    
    def _calculate_energy(self, coordinates:torch.tensor, lambda_value:float)->torch.tensor:
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
        energy_in_kJ_mol : torch.tensor
            return the energy with restraints added
        """

        
        assert(float(lambda_value) <= 1.0 and float(lambda_value) >= 0.0)

        # coordinates in nm!
        
        if self.use_pure_ani1ccx:
            _, energy_in_hartree = self.model((self.species, coordinates * nm_to_angstroms))
        else:
            _, energy_in_hartree = self.model((self.species, coordinates * nm_to_angstroms, lambda_value))
        
        # convert energy from hartrees to kJ/mol
        energy_in_kJ_mol = energy_in_hartree * hartree_to_kJ_mol
        bias_in_kJ_mol = 0.0

        for restraint in self.list_of_restraints:
            e = restraint.restraint(coordinates * nm_to_angstroms)
            if restraint.active_at_lambda == 1:
                e *= lambda_value
            elif restraint.active_at_lambda == 0:
                e *= (1 - lambda_value)
            else:
                # always on - active_at_lambda == -1
                pass 
            bias_in_kJ_mol += e
        
        energy_in_kJ_mol += bias_in_kJ_mol
        return energy_in_kJ_mol, bias_in_kJ_mol
        


    def calculate_energy(self, x:simtk.unit.quantity.Quantity, lambda_value:float = 0.0) -> simtk.unit.quantity.Quantity:
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
        E : float, unit'd 
        """

        assert(type(x) == unit.Quantity)

        coordinates = torch.tensor([x.value_in_unit(unit.nanometer)],
                                requires_grad=True, device=self.device, dtype=torch.float32)

        energy_in_kJ_mol, _ = self._calculate_energy(coordinates, lambda_value)
        return energy_in_kJ_mol.item() * unit.kilojoule_per_mole

class AlchemicalANI(torchani.models.ANI1ccx):
    
    def __init__(self, alchemical_atoms=[]):
        """Scale the contributions of alchemical atoms to the energy."""
        super().__init__()
        self.alchemical_atoms = alchemical_atoms

    def forward(self, species_coordinates, lam=1.0):
        raise (NotImplementedError)


class DirectAlchemicalANI(AlchemicalANI):
    def __init__(self, alchemical_atoms=[]):
        """Scale the direct contributions of alchemical atoms to the energy sum,
        ignoring indirect contributions
        """
        super().__init__(alchemical_atoms)


class AEVScalingAlchemicalANI(AlchemicalANI):
    def __init__(self, alchemical_atoms=[]):
        """Scale indirect contributions of alchemical atoms to the energy sum by
        interpolating neighbors' Atomic Environment Vectors.

        (Also scale direct contributions, as in DirectAlchemicalANI)
        """
        super().__init__(alchemical_atoms)


class Ensemble(torch.nn.ModuleList):
    """Compute the average output of an ensemble of modules."""

    def forward(self, species_input):
        # type: (Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]
        outputs = [x(species_input)[1].double() for x in self]
        species, _ = species_input
        energy = sum(outputs) / len(outputs)
        return species, energy 
      

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

    def __init__(self, alchemical_atoms:list, box_length:unit.Quantity=0.0 * unit.angstrom):
        """Scale the indirect contributions of alchemical atoms to the energy sum by
        linearly interpolating, for other atom i, between the energy E_i^0 it would compute
        in the _complete absence_ of the alchemical atoms, and the energy E_i^1 it would compute
        in the _presence_ of the alchemical atoms.
        (Also scale direct contributions, as in DirectAlchemicalANI)
        """
        super().__init__(alchemical_atoms)      
        self.neural_networks = load_model_ensemble(self.species, self.ensemble_prefix, self.ensemble_size)
        self.device = device
        assert(type(box_length) == unit.Quantity)
        self.box_length = box_length.value_in_unit(unit.angstrom)
            

    def forward(self, species_coordinates):

        assert(len(self.alchemical_atoms) == 1)
        alchemical_atom = self.alchemical_atoms[0]

        # LAMBDA = 1: fully interacting
        # species, AEVs of fully interacting system
        species, coordinates, lam = species_coordinates
        print(lam)
        aevs = species_coordinates[:-1]
        if self.box_length != 0.0:
            cell = torch.tensor(np.array([[self.box_length, 0.0, 0.0],[0.0,self.box_length,0.0],[0.0,0.0,self.box_length]]),
                                device=self.device, dtype=torch.float)
            aevs = aevs[0], aevs[1], cell, torch.tensor([True, True, True], dtype=torch.bool, device=self.device)

        species, aevs = self.aev_computer(aevs)


        # neural net output given these AEVs
        nn_1 = self.neural_networks((species, aevs))[1]
        E_1 = self.energy_shifter((species, nn_1))[1]
        
        # LAMBDA == 1: fully interacting
        if float(lam) == 1.0:
            E = E_1
        else:
            # LAMBDA == 0: fully removed
            # species, AEVs of all other atoms, in absence of alchemical atoms
            print(species)
            mod_species = torch.cat((species[:, :alchemical_atom],  species[:, alchemical_atom+1:]), dim=1)
            print(mod_species)
            mod_coordinates = torch.cat((coordinates[:, :alchemical_atom],  coordinates[:, alchemical_atom+1:]), dim=1) 
            mod_aevs = self.aev_computer((mod_species, mod_coordinates))[1]
            # neural net output given these modified AEVs
            nn_0 = self.neural_networks((mod_species, mod_aevs))[1]
            E_0 = self.energy_shifter((species, nn_0))[1]
            E = (lam * E_1) + ((1 - lam) * E_0)
        return species, E


class LinearAlchemicalDualTopologyANI(LinearAlchemicalANI):
   
    def __init__(self, alchemical_atoms:list, box_length:unit.Quantity=0.0 * unit.angstrom):
        """Scale the indirect contributions of alchemical atoms to the energy sum by
        linearly interpolating, for other atom i, between the energy E_i^0 it would compute
        in the _complete absence_ of the alchemical atoms, and the energy E_i^1 it would compute
        in the _presence_ of the alchemical atoms.
        (Also scale direct contributions, as in DirectAlchemicalANI)
        """

        assert(len(alchemical_atoms) == 2)

        super().__init__(alchemical_atoms, box_length)      


    def forward(self, species_coordinates):
        # for now only allow one alchemical atom

        # species, AEVs of fully interacting system
        species, coordinates, lam = species_coordinates
        aevs = species_coordinates[:-1]
        if self.box_length != 0.0:
            cell = torch.tensor(np.array([[self.box_length, 0.0, 0.0],[0.0,self.box_length,0.0],[0.0,0.0,self.box_length]]),
                                device=self.device, dtype=torch.float)
            aevs = aevs[0], aevs[1], cell, torch.tensor([True, True, True], dtype=torch.bool, device=self.device)

        # NOTE: I am not happy about this - the order at which 
        # the dummy atoms are set in alchemical_atoms determines 
        # what is real and what is dummy at lambda 1 - that seems awefully error prone
        dummy_atom_0 = self.alchemical_atoms[0]
        dummy_atom_1 = self.alchemical_atoms[1]

        # neural net output given these AEVs
        mod_species_0 = torch.cat((species[:, :dummy_atom_0],  species[:, dummy_atom_0+1:]), dim=1)
        mod_coordinates_0 = torch.cat((coordinates[:, :dummy_atom_0],  coordinates[:, dummy_atom_0+1:]), dim=1) 
        mod_aevs_0 = self.aev_computer((mod_species_0, mod_coordinates_0))[1]
        # neural net output given these modified AEVs
        nn_0 = self.neural_networks((mod_species_0, mod_aevs_0))[1]
        E_0 = self.energy_shifter((mod_species_0, nn_0))[1]
        
        # neural net output given these AEVs
        mod_species_1 = torch.cat((species[:, :dummy_atom_1],  species[:, dummy_atom_1+1:]), dim=1)
        mod_coordinates_1 = torch.cat((coordinates[:, :dummy_atom_1],  coordinates[:, dummy_atom_1+1:]), dim=1) 
        mod_aevs_1 = self.aev_computer((mod_species_1, mod_coordinates_1))[1]
        # neural net output given these modified AEVs
        nn_1 = self.neural_networks((mod_species_1, mod_aevs_1))[1]
        E_1 = self.energy_shifter((mod_species_1, nn_1))[1]

        E = (lam * E_1) + ((1 - lam) * E_0)
        return species, E