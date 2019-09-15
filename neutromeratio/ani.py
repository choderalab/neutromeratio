import os
import torchani
import torch
import numpy as np
from .constants import nm_to_angstroms, hartree_to_kJ_mol, device, platform
from simtk import unit
import simtk
from .restraints import Restraint
from ase.optimize import BFGS


class ANI1_force_and_energy(object):
    """
    Performs energy and force calculations.
    
    Parameters
    ----------
    device:
    """

    def __init__(self, 
                model:torchani.models.ANI1ccx, 
                atoms:str, 
                ):
        
        self.device = device
        self.model = model
        self.atoms = atoms
        self.species = self.model.species_to_tensor(atoms).to(device).unsqueeze(0)
        self.platform = platform
        self.use_pure_ani1ccx = False
        
        self.flat_bottom_restraint = False
        self.harmonic_restraint = False
        self.list_of_restraints = []
        self.bias = []
        # TODO: check availablity of platform

    def add_restraint(self, restraint:Restraint):
        self.list_of_restraints.append(restraint)

    def minimize(self, ani_input):
        
        calculator = self.model.ase(dtype=torch.float64)
        mol = ani_input['ase_hybrid_mol']
        mol.set_calculator(calculator)
        print("Begin minimizing...")
        opt = BFGS(mol)
        opt.run(fmax=0.001)
        ani_input['hybrid_coords'] = np.array(mol.get_positions()) * unit.angstrom

    def calculate_force(self, x:simtk.unit.quantity.Quantity, lambda_value:float) -> simtk.unit.quantity.Quantity:
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
        coordinates = torch.tensor([x.value_in_unit(unit.nanometer)],
                                requires_grad=True, device=self.device, dtype=torch.float32)

        energy_in_kJ_mol = self.calculate_energy(x, lambda_value)
        # derivative of E (in kJ/mol) w.r.t. coordinates (in nm)
        derivative = torch.autograd.grad((energy_in_kJ_mol).sum(), coordinates)[0]

        if self.platform == 'cpu':
            F = - np.array(derivative)[0]
        elif self.platform == 'cuda':
            F = - np.array(derivative.cpu())[0]
        else:
            raise RuntimeError('Platform needs to be specified. Either CPU or CUDA.')

        return F * (unit.kilojoule_per_mole / unit.nanometer), energy_in_kJ_mol.item() * unit.kilojoule_per_mole

    
    def calculate_energy(self, x:simtk.unit.quantity.Quantity, lambda_value:float) -> simtk.unit.quantity.Quantity:
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

        if self.use_pure_ani1ccx:
            _, energy_in_hartree = self.model((self.species, coordinates * nm_to_angstroms))
        else:
            _, energy_in_hartree = self.model((self.species, coordinates * nm_to_angstroms, lambda_value))
        
        # convert energy from hartrees to kJ/mol
        energy_in_kJ_mol = energy_in_hartree * hartree_to_kJ_mol

        bias_flat_bottom = 0.0
        bias_harmonic = 0.0
        bias = 0.0

        if self.flat_bottom_restraint:
            for restraint in self.list_of_restraints:
                e = restraint.flat_bottom_position_restraint(coordinates * nm_to_angstroms)
                if restraint.active_at_lambda == 1:
                    e *= lambda_value
                elif restraint.active_at_lambda == 0:
                    e *= (1 - lambda_value)
                else:
                    pass 
                bias_flat_bottom += e
                bias += e

        if self.harmonic_restraint:
            for restraint in self.list_of_restraints:
                e = restraint.harmonic_position_restraint(coordinates * nm_to_angstroms)
                if restraint.active_at_lambda == 1:
                    e *= lambda_value
                elif restraint.active_at_lambda == 0:
                    e *= (1 - lambda_value)
                else:
                    pass 
                bias_harmonic += e
                bias += e
        
        self.bias.append(bias)
        energy_in_kJ_mol += bias
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
        print('local forward?')
        print(type(energy))
        print(energy)
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
    print('local ensemble')
    for i in range(count):
        network_dir = os.path.join('{}{}'.format(prefix, i), 'networks')
        models.append(torchani.neurochem.load_model(species, network_dir))
    return Ensemble(models)


class LinearAlchemicalANI(AlchemicalANI):

    def __init__(self, alchemical_atoms:list, ani_input:dict, pbc:bool=False):
        """Scale the indirect contributions of alchemical atoms to the energy sum by
        linearly interpolating, for other atom i, between the energy E_i^0 it would compute
        in the _complete absence_ of the alchemical atoms, and the energy E_i^1 it would compute
        in the _presence_ of the alchemical atoms.
        (Also scale direct contributions, as in DirectAlchemicalANI)
        """
        super().__init__(alchemical_atoms)      
        self.neural_networks = load_model_ensemble(self.species, self.ensemble_prefix, self.ensemble_size)
        self.ani_input = ani_input
        self.device = device
        self.pbc = pbc
        if pbc:
            self.box_length = self.ani_input['box_length'].value_in_unit(unit.angstrom)
        else:
            self.box_length = 0.0 * unit.angstrom

    def forward(self, species_coordinates):
        # for now only allow one alchemical atom

        # LAMBDA = 1: fully interacting
        # species, AEVs of fully interacting system
        species, coordinates, lam = species_coordinates
        aevs = species_coordinates[:-1]
        if self.pbc:
            cell = torch.tensor(np.array([[self.box_length, 0.0, 0.0],[0.0,self.box_length,0.0],[0.0,0.0,self.box_length]]),
                                device=self.device, dtype=torch.float)
            aevs = aevs[0], aevs[1], cell, torch.tensor([True, True, True], dtype=torch.bool, device=self.device)

        species, aevs = self.aev_computer(aevs)


        # neural net output given these AEVs
        nn_1 = self.neural_networks((species, aevs))[1]
        E_1 = self.energy_shifter((species, nn_1))[1]
        
        # NOTE: this is inconsistent with the lambda definition in the staged simulation
        # LAMBDA == 1: fully interacting
        if float(lam) == 1.0:
            E = E_1
        else:
            # LAMBDA == 0: fully removed
            # species, AEVs of all other atoms, in absence of alchemical atoms
            mod_species = torch.cat((species[:, :self.alchemical_atoms[0]],  species[:, self.alchemical_atoms[0]+1:]), dim=1)
            mod_coordinates = torch.cat((coordinates[:, :self.alchemical_atoms[0]],  coordinates[:, self.alchemical_atoms[0]+1:]), dim=1) 
            mod_aevs = self.aev_computer((mod_species, mod_coordinates))[1]
            # neural net output given these modified AEVs
            nn_0 = self.neural_networks((mod_species, mod_aevs))[1]
            E_0 = self.energy_shifter((species, nn_0))[1]
            E = (lam * E_1) + ((1 - lam) * E_0)
        return species, E


class LinearAlchemicalSingleTopologyANI(LinearAlchemicalANI):
    def __init__(self, alchemical_atoms:list, ani_input:dict, device:torch.device, pbc:bool=False):
        """Scale the indirect contributions of alchemical atoms to the energy sum by
        linearly interpolating, for other atom i, between the energy E_i^0 it would compute
        in the _complete absence_ of the alchemical atoms, and the energy E_i^1 it would compute
        in the _presence_ of the alchemical atoms.
        (Also scale direct contributions, as in DirectAlchemicalANI)
        """

        assert(len(alchemical_atoms) == 2)

        super().__init__(alchemical_atoms, ani_input, device)      


    def forward(self, species_coordinates):
        # for now only allow one alchemical atom

        # species, AEVs of fully interacting system
        species, coordinates, lam = species_coordinates
        aevs = species_coordinates[:-1]
        if self.pbc:
            cell = torch.tensor(np.array([[self.box_length, 0.0, 0.0],[0.0,self.box_length,0.0],[0.0,0.0,self.box_length]]),
                                device=self.device, dtype=torch.float)
            aevs = aevs[0], aevs[1], cell, torch.tensor([True, True, True], dtype=torch.bool, device=self.device)

        dummy_atom_1 = self.alchemical_atoms[0]
        dummy_atom_2 = self.alchemical_atoms[1]

        # neural net output given these AEVs
        mod_species_1 = torch.cat((species[:, :dummy_atom_1],  species[:, dummy_atom_1+1:]), dim=1)
        mod_coordinates_1 = torch.cat((coordinates[:, :dummy_atom_1],  coordinates[:, dummy_atom_1+1:]), dim=1) 
        mod_aevs_1 = self.aev_computer((mod_species_1, mod_coordinates_1))[1]
        # neural net output given these modified AEVs
        nn_0 = self.neural_networks((mod_species_1, mod_aevs_1))[1]
        E_0 = self.energy_shifter((mod_species_1, nn_0))[1]
        
        # neural net output given these AEVs
        mod_species_2 = torch.cat((species[:, :dummy_atom_2],  species[:, dummy_atom_2+1:]), dim=1)
        mod_coordinates_2 = torch.cat((coordinates[:, :dummy_atom_2],  coordinates[:, dummy_atom_2+1:]), dim=1) 
        mod_aevs_2 = self.aev_computer((mod_species_2, mod_coordinates_2))[1]
        # neural net output given these modified AEVs
        nn_1 = self.neural_networks((mod_species_2, mod_aevs_2))[1]
        E_1 = self.energy_shifter((mod_species_2, nn_1))[1]

        E = (lam * E_1) + ((1 - lam) * E_0)
        return species, E