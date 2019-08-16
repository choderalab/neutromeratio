import os
import torchani
import torch
import numpy as np
from .constants import nm_to_angstroms, hartree_to_kJ_mol
from simtk import unit
import simtk
from rdkit import Chem
import logging
from openeye import oechem
import openmoltools as omtff
from io import StringIO
#from lxml import etree
import simtk.openmm.app as app
from .restraints import FlatBottomRestraint
from .mcmc import reduced_pot
import neutromeratio

gaff_default = os.path.join("../data/gaff2.xml")
logger = logging.getLogger(__name__)



class ANI1_force_and_energy(object):
    """
    Performs energy and force calculations.
    
    Parameters
    ----------
    device:
    """

    def __init__(self, device:torch.device, model:torchani.models.ANI1ccx, atom_list:list, platform:str, tautomer_transformation:dict={}):
        
        self.device = device
        self.model = model
        self.species = model.species_to_tensor(atom_list).to(device).unsqueeze(0)
        self.atom_list = atom_list
        self.platform = platform
        self.lambda_value = 1.0
        self.bias = []
        self.tautomer_transformation = tautomer_transformation
        self.restrain_acceptor = False
        self.restrain_donor = False

        # TODO: check availablity of platform


    def compute_restraint(self, coordinates):
        if self.restrain_acceptor or self.restrain_donor:
            if self.restrain_acceptor:
                heavy_atom_idx = self.tautomer_transformation['acceptor_idx']
            elif self.restrain_donor:
                heavy_atom_idx = self.tautomer_transformation['donor_idx']

            restraint = FlatBottomRestraint(
                heavy_atom_index=heavy_atom_idx,
                hydrogen_index=self.tautomer_transformation['hydrogen_idx'])
            bias = restraint.forward(coordinates)
        else:
            bias = 0
        return bias

    def calculate_force(self, x:simtk.unit.quantity.Quantity)->simtk.unit.quantity.Quantity:
        """
        Given a coordinate set the forces with respect to the coordinates are calculated.
        
        Parameters
        ----------
        x : array of floats, unit'd (distance unit)
            initial configuration

        Returns
        -------
        F : float, unit'd
            
        """

        assert(type(x) == unit.Quantity)
        coordinates = torch.tensor([x.value_in_unit(unit.nanometer)],
                                requires_grad=True, device=self.device, dtype=torch.float32)


        if type(self.model) is torchani.models.ANI1ccx:
            _, energy_in_hartree = self.model((self.species, coordinates * nm_to_angstroms))
        elif type(self.model) is neutromeratio.ani.LinearAlchemicalANI:
            _, energy_in_hartree = self.model((self.species, coordinates * nm_to_angstroms, self.lambda_value))
        else:
            raise NotImplementedError('Only Ani1ccx or AlchemicalAni models are allowed.')
        # convert energy from hartrees to kJ/mol
        energy_in_kJ_mol = energy_in_hartree * hartree_to_kJ_mol

        bias = self.compute_restraint(coordinates)
        self.bias.append(bias)
        energy_in_kJ_mol = energy_in_kJ_mol + bias

        # derivative of E (in kJ/mol) w.r.t. coordinates (in nm)
        derivative = torch.autograd.grad((energy_in_kJ_mol).sum(), coordinates)[0]

        if self.platform == 'cpu':
            F = - np.array(derivative)[0]
        elif self.platform == 'cuda':
            F = - np.array(derivative.cpu())[0]
        else:
            raise RuntimeError('Platform needs to be specified. Either CPU or CUDA.')

        return F * (unit.kilojoule_per_mole / unit.nanometer)

    
    def calculate_energy(self, x:simtk.unit.quantity.Quantity)->simtk.unit.quantity.Quantity:
        """
        Given a coordinate set (x) the energy is calculated in kJ/mol.

        Parameters
        ----------
        x : array of floats, unit'd (distance unit)
            initial configuration

        Returns
        -------
        E : float, unit'd 
        """

        assert(type(x) == unit.Quantity)

        coordinates = torch.tensor([x.value_in_unit(unit.nanometer)],
                                requires_grad=True, device=self.device, dtype=torch.float32)

        if type(self.model) is torchani.models.ANI1ccx:
            _, energy_in_hartree = self.model((self.species, coordinates * nm_to_angstroms))
        elif type(self.model) is neutromeratio.ani.LinearAlchemicalANI:
            _, energy_in_hartree = self.model((self.species, coordinates * nm_to_angstroms, self.lambda_value))
        else:
            raise NotImplementedError('Only Ani1ccx or AlchemicalAni models are allowed.')
        # convert energy from hartrees to kJ/mol
        energy_in_kJ_mol = energy_in_hartree * hartree_to_kJ_mol

        bias = self.compute_restraint(x)
        energy_in_kJ_mol = energy_in_kJ_mol + bias
        return energy_in_kJ_mol.item() * unit.kilojoule_per_mole


class AlchemicalANI(torchani.models.ANI1ccx):
    
    def __init__(self, alchemical_atoms=[0]):
        """Scale the contributions of alchemical atoms to the energy."""
        super().__init__()
        self.alchemical_atoms = alchemical_atoms
        self.test = True

    def forward(self, species_coordinates, lam=1.0):
        raise (NotImplementedError)


class DirectAlchemicalANI(AlchemicalANI):
    def __init__(self, alchemical_atoms=[0]):
        """Scale the direct contributions of alchemical atoms to the energy sum,
        ignoring indirect contributions
        """
        super().__init__(alchemical_atoms)


class AEVScalingAlchemicalANI(AlchemicalANI):
    def __init__(self, alchemical_atoms=[0]):
        """Scale indirect contributions of alchemical atoms to the energy sum by
        interpolating neighbors' Atomic Environment Vectors.

        (Also scale direct contributions, as in DirectAlchemicalANI)
        """
        super().__init__(alchemical_atoms)



class DoubleAniModel(torchani.nn.ANIModel):

    def forward(self, species_aev):
        # change dtype
        species, aev = species_aev
        species_ = species.flatten()
        present_species = torchani.utils.present_species(species)
        aev = aev.flatten(0, 1)

        output = torch.full_like(species_, self.padding_fill,
                                dtype=torch.float32)
        for i in present_species:
            mask = (species_ == i)
            input_ = aev.index_select(0, mask.nonzero().squeeze())
            output.masked_scatter_(mask, self[i](input_).squeeze())
        
        output = output.view_as(species)
        return species, self.reducer(output.double(), dim=1)
        

class LinearAlchemicalANI(AlchemicalANI):
    def __init__(self, alchemical_atoms):
        """Scale the indirect contributions of alchemical atoms to the energy sum by
        linearly interpolating, for other atom i, between the energy E_i^0 it would compute
        in the _complete absence_ of the alchemical atoms, and the energy E_i^1 it would compute
        in the _presence_ of the alchemical atoms.
        (Also scale direct contributions, as in DirectAlchemicalANI)
        """

        super().__init__(alchemical_atoms)      
        self.neural_networks = self._load_model_ensemble(self.species, self.ensemble_prefix, self.ensemble_size)

    def forward(self, species_coordinates):
        # for now only allow one alchemical atom

        # LAMBDA = 1: fully interacting
        # species, AEVs of fully interacting system
        species, coordinates, lam = species_coordinates
        species, aevs = self.aev_computer(species_coordinates[:-1])
        # neural net output given these AEVs
        nn_1 = self.neural_networks((species, aevs))[1]
        E_1 = self.energy_shifter((species, nn_1))[1]
        # LAMBDA = 0: fully removed
        # species, AEVs of all other atoms, in absence of alchemical atoms
        mod_species = torch.cat((species[:, :self.alchemical_atoms],  species[:, self.alchemical_atoms+1:]), dim=1)
        mod_coordinates = torch.cat((coordinates[:, :self.alchemical_atoms],  coordinates[:, self.alchemical_atoms+1:]), dim=1) 
        mod_aevs = self.aev_computer((mod_species, mod_coordinates))[1]
        # neural net output given these modified AEVs
        nn_0 = self.neural_networks((mod_species, mod_aevs))[1]
        E_0 = self.energy_shifter((species, nn_0))[1]

        return species, (lam * E_1) + ((1 - lam) * E_0)


    def _load_model_ensemble(self, species, prefix, count):
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
            models.append(self._load_model(species, network_dir))
        return torchani.nn.Ensemble(models)

    def _load_model(self, species, dir_):
        """Returns an instance of :class:`torchani.ANIModel` loaded from
        NeuroChem's network directory.
        Arguments:
            species (:class:`collections.abc.Sequence`): Sequence of strings for
                chemical symbols of each supported atom type in correct order.
            dir_ (str): String for directory storing network configurations.
        """
        models = []
        for i in species:
            filename = os.path.join(dir_, 'ANN-{}.nnf'.format(i))
            models.append(torchani.neurochem.load_atomic_network(filename))
        return DoubleAniModel(models)

# TODO: implement a base class Energy 
# TODO: openMM smirnoff/gaff energy subclass
# TODO: psi4 energy
