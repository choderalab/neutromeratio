import logging, copy, os, random
import mdtraj as md
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from .vis import display_mol
from .restraints import Restraint
from .ani import ANI1_force_and_energy
from simtk import unit
from .utils import write_pdb
import numpy as np
from ase import Atom, Atoms
from .constants import device, platform
from .mcmc import MC_Mover
import torch, torchani

logger = logging.getLogger(__name__)

class Tautomer(object):

    def __init__(self, name:str, intial_state_mol:Chem.Mol, final_state_mol:Chem.Mol, nr_of_conformations:int=1):

        self.name = name
        self.nr_of_conformations = nr_of_conformations
        assert(type(intial_state_mol) == Chem.Mol)
        assert(type(final_state_mol) == Chem.Mol)

        self.intial_state_mol:Chem.Mol = intial_state_mol
        self.final_state_mol:Chem.Mol = final_state_mol

        intial_state_ani_input = self._from_mol_to_ani_input(self.intial_state_mol)
        self.intial_state_ligand_atoms = intial_state_ani_input['ligand_atoms']
        self.intial_state_ligand_coords = intial_state_ani_input['ligand_coords']
        self.intial_state_ligand_topology:md.Topology = intial_state_ani_input['ligand_topology']
        self.intial_state_ase_mol:Atoms = intial_state_ani_input['ase_mol']

        final_state_ani_input = self._from_mol_to_ani_input(self.final_state_mol)
        self.final_state_ligand_atoms = final_state_ani_input['ligand_atoms']
        self.final_state_ligand_coords = final_state_ani_input['ligand_coords']
        self.final_state_ligand_topology:md.Topology = final_state_ani_input['ligand_topology']
        self.final_state_ase_mol:Atoms = final_state_ani_input['ase_mol']

        # attributes for the equilibrium protocol
        # these values are set by perform_tautomer_transformation_forward
        # or perform_tautomer_transformation_reverse and correspond either
        # to the idx of initial in case of forward or final in case of reverse
        self.hybrid_hydrogen_idx_at_donor_heavy_atom:int = -1
        self.hybrid_hydrogen_idx_at_acceptor_heavy_atom:int = -1
        self.hybrid_coords:list = []
        self.hybrid_ase_mol:Atoms = None
        self.hybrid_topology:md.Topology = None
        
        # attributes for the NCMC protocol
        self.heavy_atom_hydrogen_donor_idx:int = -1
        self.hydrogen_idx:int = -1
        self.heavy_atom_hydrogen_acceptor_idx:int = -1
        self.ncmc_restraints:list = []


    def perform_tautomer_transformation_forward(self):
        """
        Returns the atom index of the hydrogen donor atom and hydrogen atom that moves.
        """

        m1 = copy.deepcopy(self.intial_state_mol)
        m2 = copy.deepcopy(self.final_state_mol)
        self._perform_tautomer_transformation(m1, m2)
        self._generate_hybrid_structure(self.intial_state_ligand_atoms, None, self.intial_state_ligand_coords[0], self.intial_state_ligand_topology)

    def perform_tautomer_transformation_reverse(self):
        """
        Returns the atom index of the hydrogen donor atom and hydrogen atom that moves.
        This index is consistent with the indexing of m1.
        Parameters
        ----------
        m1: rdkit mol object
        m2: rdkit mol object
        
        Returns
        -------
        { 'donor_idx': donor, 'hydrogen_idx' : hydrogen_idx_that_moves, 'acceptor_idx' : acceptor}
        """

        m1 = copy.deepcopy(self.final_state_mol)
        m2 = copy.deepcopy(self.intial_state_mol)
        self._perform_tautomer_transformation(m1, m2)
        self._generate_hybrid_structure(self.final_state_ligand_atoms, None, self.final_state_ligand_coords[0], self.final_state_ligand_topology)


    def _from_mol_to_ani_input(self, mol:Chem.Mol):
        """
        Generates atom_list and coord_list entries from rdkit mol.
        """
        
        # generate atom list
        atom_list = []
        for a in mol.GetAtoms():
            atom_list.append(a.GetSymbol())

        # generate conformations
        mol = self._generate_conformations_from_mol(mol, self.nr_of_conformations)

        # generate coord list
        coord_list = []

        # add conformations to coord_list
        for conf_idx in range(mol.GetNumConformers()):
            tmp_coord_list = []
            for a in mol.GetAtoms():
                pos = mol.GetConformer(conf_idx).GetAtomPosition(a.GetIdx())
                tmp_coord_list.append([pos.x, pos.y, pos.z])
            tmp_coord_list = np.array(tmp_coord_list) * unit.angstrom
            coord_list.append(tmp_coord_list)

        # get mdtraj topology
        n = random.random()
        # TODO: use tmpfile for this https://stackabuse.com/the-python-tempfile-module/ or io.StringIO
        _ = write_pdb(mol, f"tmp{n:0.9f}.pdb")
        topology = md.load(f"tmp{n:0.9f}.pdb").topology
        os.remove(f"tmp{n:0.9f}.pdb")
        
        ani_input =  {'ligand_atoms' : ''.join(atom_list), 
                'ligand_coords' : coord_list, 
                'ligand_topology' : topology
                }

        # generate ONE ASE object
        ase_atom_list = []
        for e, c in zip(ani_input['ligand_atoms'], coord_list[0]):
            c_list = (c[0].value_in_unit(unit.angstrom), c[1].value_in_unit(unit.angstrom), c[2].value_in_unit(unit.angstrom)) 
            ase_atom_list.append(Atom(e, c_list))
        mol = Atoms(ase_atom_list)
        ani_input['ase_mol'] = mol
        return ani_input


    def _generate_conformations_from_mol(self, mol:Chem.Mol, nr_of_conformations:int):
        """
        Generates a rdkit molecule from a SMILES string, generates conformations and generates a dictionary representation of it.
        
        Keyword arguments:
        input_smi: SMILES string
        nr_of_conformations: int
        """  

        charge = 0
        for at in mol.GetAtoms():
            if at.GetFormalCharge() != 0:
                charge += int(at.GetFormalCharge())          

        if charge != 0:
            print(Chem.MolToSmiles(mol))
            raise NotImplementedError('Charged system')

        Chem.rdmolops.RemoveStereochemistry(mol)
        Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)

        mol.SetProp("smiles", Chem.MolToSmiles(mol))
        mol.SetProp("charge", str(charge))
        
        mol.SetProp("name", str(self.name))

        # generate numConfs for the smiles string 
        Chem.rdDistGeom.EmbedMultipleConfs(mol, numConfs=nr_of_conformations, enforceChirality=False, pruneRmsThresh=0.1)
        # aligne them and minimize
        AllChem.AlignMolConformers(mol)
        AllChem.MMFFOptimizeMoleculeConfs(mol)
        return mol


    def _perform_tautomer_transformation(self, m1:Chem.Mol, m2:Chem.Mol):
        """
        Returns the atom index of the hydrogen donor atom and hydrogen atom that moves.
        This index is consistent with the indexing of m1.
        Parameters
        ----------
        m1: rdkit mol object
        m2: rdkit mol object
        
        Returns
        -------
        { 'donor_idx': donor, 'hydrogen_idx' : hydrogen_idx_that_moves, 'acceptor_idx' : acceptor}
        """

        # find substructure and generate mol from substructure
        sub_m = rdFMCS.FindMCS([m1, m2], bondCompare=Chem.rdFMCS.BondCompare.CompareOrder.CompareAny)
        mcsp = Chem.MolFromSmarts(sub_m.smartsString, False)

        # the order of the substructure lists are the same for both 
        # substructure matches => substructure_idx_m1[i] = substructure_idx_m2[i]
        substructure_idx_m1 = m1.GetSubstructMatch(mcsp)
        substructure_idx_m2 = m2.GetSubstructMatch(mcsp)

        #get idx of hydrogen that moves to new position
        hydrogen_idx_that_moves = -1
        atoms = '' # atom element string
        for a in m1.GetAtoms():
            atoms += str(a.GetSymbol())

            if a.GetIdx() not in substructure_idx_m1:
                logger.info('Index of atom that moves: {}.'.format(a.GetIdx()))
                hydrogen_idx_that_moves = a.GetIdx()

        # get idx of connected heavy atom which is the donor atom
        # there can only be one neighbor, therefor it is valid to take the first neighbor of the hydrogen
        donor = int(m1.GetAtomWithIdx(hydrogen_idx_that_moves).GetNeighbors()[0].GetIdx())
        logger.info('Index of atom that donates hydrogen: {}'.format(donor))

        logging.debug(substructure_idx_m1)
        logging.debug(substructure_idx_m2)
        for i in range(len(substructure_idx_m1)):
            a1 = m1.GetAtomWithIdx(substructure_idx_m1[i])
            if a1.GetSymbol() != 'H':
                a2 = m2.GetAtomWithIdx(substructure_idx_m2[i])
                # get acceptor - there are two heavy atoms that have 
                # not the same number of neighbors
                a1_neighbors = a1.GetNeighbors()
                a2_neighbors = a2.GetNeighbors()
                acceptor_count = 0
                if (len(a1_neighbors)) != (len(a2_neighbors)):
                    # we are only interested in the one that is not already the donor
                    if substructure_idx_m1[i] == donor:
                        continue
                    acceptor = substructure_idx_m1[i]
                    logger.info('Index of atom that accepts hydrogen: {}'.format(acceptor))
                    acceptor_count += 1
                    if acceptor_count > 1:
                        raise RuntimeError('There are too many potential acceptor atoms.')

        AllChem.Compute2DCoords(m1)
        display_mol(m1)
        
        r1 = Restraint( sigma=0.1 * unit.angstrom, atom_i_idx=donor, atom_j_idx=hydrogen_idx_that_moves, atoms=atoms, active_at_lambda=1)
        r2 = Restraint( sigma=0.1 * unit.angstrom, atom_i_idx=acceptor, atom_j_idx=hydrogen_idx_that_moves, atoms=atoms, active_at_lambda=0)

        self.heavy_atom_hydrogen_donor_idx = donor
        self.hydrogen_idx = hydrogen_idx_that_moves
        self.heavy_atom_hydrogen_acceptor_idx = acceptor
        self.ncmc_restraints = [r1,r2]



    def _generate_hybrid_structure(self, ligand_atoms:str, ase_mol:Atoms, ligand_coords, ligand_topology:md.Topology):
        """
        Generates a hybrid structure between two tautomers. The heavy atom frame is kept but a
        hydrogen is added to the tautomer acceptor heavy atom. 
        """

        # add hybrid atoms
        hybrid_atoms = ligand_atoms + 'H'

        # generate 3D coordinates for hybrid atom
        model = torchani.models.ANI1ccx()
        model = model.to(device)

        energy_function = ANI1_force_and_energy(
                                            model = model,
                                            atoms = hybrid_atoms,
                                            mol = None
                                            )
        
        self.hybrid_atoms = hybrid_atoms
        energy_function.use_pure_ani1ccx = True
        # generate MC mover to get new hydrogen position
        hydrogen_mover = MC_Mover(self.heavy_atom_hydrogen_donor_idx, 
                                self.hydrogen_idx, 
                                self.heavy_atom_hydrogen_acceptor_idx,
                                ligand_atoms)


        min_e = 100 * unit.kilocalorie_per_mole
        min_coordinates = None

        for _ in range(500):
            hybrid_coord = hydrogen_mover._move_hydrogen_to_acceptor_idx(ligand_coords, override=False)
            e = energy_function.calculate_energy(hybrid_coord, lambda_value=1.0)
            if e < min_e:
                min_e = e
                min_coordinates = hybrid_coord 
        

        # add indixes for the hydrogens that are active at labda 0 (donor_hydrogen_idx)
        # and at lambda 1 (acceptor_hydrogen_idx)
        self.hybrid_hydrogen_idx_at_donor_heavy_atom = self.hydrogen_idx
        self.hybrid_hydrogen_idx_at_acceptor_heavy_atom = len(hybrid_atoms) -1
        self.hybrid_coords = min_coordinates

        # generate hybrid restraints
        # donor restraints are active at lambda = 1 because then donor hydrogen is turend off and vica versa
        r1 = Restraint(0.1 * unit.angstrom, self.hybrid_hydrogen_idx_at_donor_heavy_atom, self.heavy_atom_hydrogen_donor_idx, atoms=hybrid_atoms, active_at_lambda=1)
        r2 = Restraint(0.1 * unit.angstrom, self.hybrid_hydrogen_idx_at_acceptor_heavy_atom, self.heavy_atom_hydrogen_acceptor_idx, atoms=hybrid_atoms, active_at_lambda=0)
        self.hybrid_restraints = [r1, r2]

        # add to mdtraj ligand topology a new hydrogen
        hybrid_topology = copy.deepcopy(ligand_topology)
        dummy_atom = hybrid_topology.add_atom('H', md.element.hydrogen, hybrid_topology.residue(-1))
        hybrid_topology.add_bond(hybrid_topology.atom(self.heavy_atom_hydrogen_acceptor_idx), dummy_atom)
        # save new top in ani_input

        self.hybrid_topology = hybrid_topology

        # generate an ASE topology for the hybrid mol to minimze later 
        atom_list = []
        for e, c in zip(self.hybrid_atoms, self.hybrid_coords):
            c_list = (c[0].value_in_unit(unit.angstrom), c[1].value_in_unit(unit.angstrom), c[2].value_in_unit(unit.angstrom)) 
            atom_list.append(Atom(e, c_list))
        mol = Atoms(atom_list)
        self.hybrid_ase_mol = mol


