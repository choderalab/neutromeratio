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
import torchani.models.ANI1ccx

gaff_default = os.path.join("../data/gaff2.xml")
logger = logging.getLogger(__name__)


class ANI1_force_and_energy(object):
    """
    Performs energy and force calculations.
    
    Parameters
    ----------
    device:
    """

    def __init__(self, device:torch.device, model:torchani.models.ANI1ccx, species:torch.Tensor, platform:str):
        self.device = device
        self.model = model
        self.species = species
        self.platform = platform

        # TODO: check availablity of platform

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

        _, energy_in_hartree = self.model((self.species, coordinates * nm_to_angstroms))

        # convert energy from hartrees to kJ/mol
        energy_in_kJ_mol = energy_in_hartree * hartree_to_kJ_mol

        # derivative of E (in kJ/mol) w.r.t. coordinates (in nm)
        derivative = torch.autograd.grad((energy_in_kJ_mol).sum(), coordinates)[0]

        if self.platform == 'cpu':
            F = - np.array(derivative)[0]
        elif self.platform == 'cuda':
            F = - np.array(derivative.cpu())[0]
        else:
            raise RuntimeError('Platform needs to be specified. Either CPU or CUDA.')

        return F * (unit.kilojoule_per_mole / unit.nanometer)

    
    def calculate_energy(self, x:simtk.unit.quantity.Quantity, lambda_list:list)->simtk.unit.quantity.Quantity:
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

        coordinates = torch.tensor([x.value_in_unit(unit.angstrom)],
                                requires_grad=True, device=self.device, dtype=torch.float32)

        _, energy_in_hartree = self.model((self.species, coordinates))

        # convert energy from hartrees to kJ/mol
        energy_in_kJ_mol = energy_in_hartree * hartree_to_kJ_mol

        return energy_in_kJ_mol.item() * unit.kilojoule_per_mole




class AlchemicalANI(torchani.models.ANI1ccx):
    def __init__(self, alchemical_atoms=[0]):
        """Scale the contributions of alchemical atoms to the energy."""
        super().__init__()
        self.alchemical_atoms = alchemical_atoms

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


class LinearAlchemicalANI(AlchemicalANI):
    def __init__(self, alchemical_atoms=[0]):
        """Scale the indirect contributions of alchemical atoms to the energy sum by
        linearly interpolating, for other atom i, between the energy E_i^0 it would compute
        in the _complete absence_ of the alchemical atoms, and the energy E_i^1 it would compute
        in the _presence_ of the alchemical atoms.

        (Also scale direct contributions, as in DirectAlchemicalANI)
        """
        super().__init__(alchemical_atoms)





















































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
