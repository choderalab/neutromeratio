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
from lxml import etree
import simtk.openmm.app as app
from .restraints import flat_bottom_position_restraint, harmonic_position_restraint
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

    def __init__(self, 
                device:torch.device, 
                model:torchani.models.ANI1ccx, 
                atom_list:list, 
                platform:str, 
                tautomer_transformation:dict={},
                pbc:bool = False,
                ):
        
        self.device = device
        self.model = model
        self.species = model.species_to_tensor(atom_list).to(device).unsqueeze(0)
        self.atom_list = atom_list
        self.platform = platform
        self.lambda_value = 1.0
        self.bias_harmonic = []
        self.bias_flat_bottom = []
        self.bias_applied = []
        self.tautomer_transformation = tautomer_transformation
        self.restrain_acceptor = False
        self.restrain_donor = False

        # TODO: check availablity of platform

    def calculate_force(self, x:simtk.unit.quantity.Quantity) -> simtk.unit.quantity.Quantity:
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


        _, energy_in_hartree = self.model((self.species, coordinates * nm_to_angstroms, self.lambda_value))
        # convert energy from hartrees to kJ/mol
        energy_in_kJ_mol = energy_in_hartree * hartree_to_kJ_mol

        if self.restrain_acceptor or self.restrain_donor:
            bias_flat_bottom = flat_bottom_position_restraint(coordinates, self.tautomer_transformation, self.atom_list, restrain_acceptor = self.restrain_acceptor, restrain_donor = self.restrain_donor)
            bias_harmonic = harmonic_position_restraint(coordinates, self.tautomer_transformation, self.atom_list, restrain_acceptor = self.restrain_acceptor, restrain_donor = self.restrain_donor)
            bias = (bias_flat_bottom * self.lambda_value) + ((1 - self.lambda_value) * bias_harmonic)

            self.bias_flat_bottom.append(bias_flat_bottom)
            self.bias_harmonic.append(bias_harmonic)            
            self.bias_applied.append(bias)
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

    
    def calculate_energy(self, x:simtk.unit.quantity.Quantity) -> simtk.unit.quantity.Quantity:
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

        _, energy_in_hartree = self.model((self.species, coordinates * nm_to_angstroms, self.lambda_value))
        # convert energy from hartrees to kJ/mol
        energy_in_kJ_mol = energy_in_hartree * hartree_to_kJ_mol

        if self.restrain_acceptor or self.restrain_donor:
            bias_flat_bottom = flat_bottom_position_restraint(coordinates, self.tautomer_transformation, self.atom_list, restrain_acceptor = self.restrain_acceptor, restrain_donor = self.restrain_donor)
            bias_harmonic = harmonic_position_restraint(coordinates, self.tautomer_transformation, self.atom_list, restrain_acceptor = self.restrain_acceptor, restrain_donor = self.restrain_donor)           
            bias = (bias_flat_bottom * self.lambda_value) + ((1 - self.lambda_value) * bias_harmonic)
            energy_in_kJ_mol = energy_in_kJ_mol + bias

        return energy_in_kJ_mol.item() * unit.kilojoule_per_mole

class AlchemicalANI(torchani.models.ANI1ccx):
    
    def __init__(self, alchemical_atom=0):
        """Scale the contributions of alchemical atoms to the energy."""
        super().__init__()
        self.alchemical_atom = alchemical_atom

    def forward(self, species_coordinates, lam=1.0):
        raise (NotImplementedError)


class DirectAlchemicalANI(AlchemicalANI):
    def __init__(self, alchemical_atom=0):
        """Scale the direct contributions of alchemical atoms to the energy sum,
        ignoring indirect contributions
        """
        super().__init__(alchemical_atom)


class AEVScalingAlchemicalANI(AlchemicalANI):
    def __init__(self, alchemical_atom=0):
        """Scale indirect contributions of alchemical atoms to the energy sum by
        interpolating neighbors' Atomic Environment Vectors.

        (Also scale direct contributions, as in DirectAlchemicalANI)
        """
        super().__init__(alchemical_atom)



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
    def __init__(self, alchemical_atom:int, ani_input:dict, device:torch.device):
        """Scale the indirect contributions of alchemical atoms to the energy sum by
        linearly interpolating, for other atom i, between the energy E_i^0 it would compute
        in the _complete absence_ of the alchemical atoms, and the energy E_i^1 it would compute
        in the _presence_ of the alchemical atoms.
        (Also scale direct contributions, as in DirectAlchemicalANI)
        """

        super().__init__(alchemical_atom)      
        self.neural_networks = self._load_model_ensemble(self.species, self.ensemble_prefix, self.ensemble_size)
        self.ani_input = ani_input
        self.device = device
        if 'box_length' in ani_input:
            self.pbc = True
        else:
            self.pbc = False
                        

    def forward(self, species_coordinates):
        # for now only allow one alchemical atom

        # LAMBDA = 1: fully interacting
        # species, AEVs of fully interacting system
        species, coordinates, lam = species_coordinates
        aevs = species_coordinates[:-1]
        if self.pbc:
            box_length = self.ani_input['box_length'].value_in_unit(unit.angstrom)
            cell = torch.tensor(np.array([[box_length, 0.0, 0.0],[0.0,box_length,0.0],[0.0,0.0,box_length]]),
                                device=self.device, dtype=torch.double)
            aevs = aevs[0], aevs[1], cell, torch.tensor([True, True, True], dtype=torch.bool,device=self.device)

        species, aevs = self.aev_computer(aevs)
            

        # neural net output given these AEVs
        nn_1 = self.neural_networks((species, aevs))[1]
        E_1 = self.energy_shifter((species, nn_1))[1]
        
        if float(lam) == 1.0:
            E = E_1
        else:
            # LAMBDA = 0: fully removed
            # species, AEVs of all other atoms, in absence of alchemical atoms
            mod_species = torch.cat((species[:, :self.alchemical_atom],  species[:, self.alchemical_atom+1:]), dim=1)
            mod_coordinates = torch.cat((coordinates[:, :self.alchemical_atom],  coordinates[:, self.alchemical_atom+1:]), dim=1) 
            mod_aevs = self.aev_computer((mod_species, mod_coordinates))[1]
            # neural net output given these modified AEVs
            nn_0 = self.neural_networks((mod_species, mod_aevs))[1]
            E_0 = self.energy_shifter((species, nn_0))[1]
            E = (lam * E_1) + ((1 - lam) * E_0)
        return species, E


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

# class Energy(object):

#     def __init__(self, smiles:str):

#         self.smiles = smiles
#         # prepare openeye mol
#         m = omtff.openeye.smiles_to_oemol(smiles)
#         m = omtff.openeye.generate_conformers(m, max_confs=1)
#         m.SetTitle("m1")
#         # write pdb
#         ofs.open(f"tmp.pdb")
#         ofs.SetFormat(oechem.OEFormat_PDB)
#         oechem.OEWriteMolecule(ofs, m)

#         # generate force field
#         ffxml = omtff.forcefield_generators.generateForceFieldFromMolecules(
#         [m], 
#         normalize=False, 
#         gaff_version='gaff2'
#         )


#         # create bond and hydrogen definitions
#         bxml = self._create_bond_definitions(StringIO(ffxml), f"t{t_id:02d}")
#         bxml = StringIO(etree.tostring(bxml).decode("utf-8"))

#         app.Topology.loadBondDefinitions(bxml)

#         hxml = self._create_hydrogen_definitions(etree.fromstring(ffxml))
#         hxml = StringIO(etree.tostring(hxml).decode("utf-8"))
        
#         app.Modeller.loadHydrogenDefinitions(hxml)
#         pdb = app.PDBFile("tmp.pdb")
#         modeller = app.Modeller(pdb.topology, pdb.positions)
#         modeller.addHydrogens()

#         integrator = mm.LangevinIntegrator(
#                         300*unit.kelvin,       # Temperature of heat bath
#                         1.0/unit.picoseconds,  # Friction coefficient
#                         2.0*unit.femtoseconds, # Time step
#         )

#         system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
#         sim = app.Simulation(modeller.topology, system, integrator)
#         sim.context.setPositions(modeller.positions)




#     def _create_hydrogen_definitions(
#         self,
#         xmltree
#     ):
#         """
#         Generates hydrogen definitions for a small molecule residue template.

#         """
#         hydrogen_definitions_tree = etree.fromstring("<Residues/>")
#         for residue in xmltree.xpath("Residues/Residue"):
#             hydrogen_file_residue = etree.fromstring("<Residue/>")
#             hydrogen_file_residue.set("name", residue.get("name"))
#             # enumerate hydrogens in this list
#             hydrogens = list()
#             # Loop through atoms to find all hydrogens
#             for bond in residue.xpath("Bond"):
#                 atomname1 = bond.get("atomName1")
#                 atomname2 = bond.get("atomName2")
#                 if 'H' in atomname1:
#                     # H is the first, parent is the second atom
#                     hydrogens.append(tuple([atomname1, atomname2]))
#                 elif 'H' in atomname2:
#                     # H is the second, parent is the first atom
#                     hydrogens.append(tuple([atomname2, atomname1]))

#             # Loop through all hydrogens, and create definitions
#             for name, parent in hydrogens:
#                 h_xml = etree.fromstring("<H/>")
#                 h_xml.set("name", name)
#                 h_xml.set("parent", parent)
#                 hydrogen_file_residue.append(h_xml)
#             hydrogen_definitions_tree.append(hydrogen_file_residue)
#         # Write output
#         return hydrogen_definitions_tree



#     def _create_bond_definitions(
#         self,
#         inputfile: str,
#         residue_name : str = None
#         ):
#         """
#         Generates bond definitions for a small molecule template to subsequently load 
#         the bond definitions in the topology object. BE CAREFULL: The residue name 
#         of the pdb file must match the residue name in the bxml file.

#         Parameters
#         ----------
#         inputfile - a forcefield XML file defined using Gaff atom types
#         """

#         xmltree = etree.parse(
#             inputfile, etree.XMLParser(remove_blank_text=True, remove_comments=True)
#         )
#         # Output tree
#         bond_definitions_tree = etree.fromstring("<Residues/>")
#         bonds = set()

#         for residue in xmltree.xpath("Residues/Residue"):
#             # Loop through all bonds
#             bond_file_residue = etree.fromstring("<Residue/>")
#             bond_file_residue.set("name", f"{residue_name}")
#             for bond in residue.xpath("Bond"):
#                 atomname1 = bond.get("atomName1")
#                 atomname2 = bond.get("atomName2")
#                 if atomname1.startswith('H') or atomname2.startswith('H'):
#                     continue
#                 bonds.add(tuple([atomname1, atomname2]))
            
#             for a1, a2 in bonds:
#                 b_xml = etree.fromstring("<Bond/>")
#                 b_xml.set("from", a1)
#                 b_xml.set("to", a2)
#                 bond_file_residue.append(b_xml)
#             bond_definitions_tree.append(bond_file_residue)

#         return bond_definitions_tree
