import logging, copy, os, random
import mdtraj as md
import parmed as pm
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from .vis import display_mol
from .restraints import FlatBottomRestraint, FlatBottomRestraintToCenter, CenterOfMassRestraint
from .ani import ANI1_force_and_energy
from simtk import unit
from .utils import write_pdb
import numpy as np
from ase import Atom, Atoms
from .constants import device, platform
from .mcmc import MC_Mover
import torch, torchani
from .constants import temperature, gas_constant, kT
from scipy.special import logsumexp
from pdbfixer import PDBFixer
from simtk.openmm import Vec3
import networkx as nx

logger = logging.getLogger(__name__)

class Tautomer(object):
    """
    A tautomer object that holds two tautomeric forms of a single molecule.
    
    Parameters
    ----------
    name: str
        the name of the small molecule
    initial_state_mol: rdkit.Chem.Mol
        initial state mol
    final_state_mol: rdkit.Chem.Mol
        final state mol
    nr_of_conformations : int
        nr of conformations that are calculated
    """

    def __init__(self, name:str, initial_state_mol:Chem.Mol, final_state_mol:Chem.Mol, nr_of_conformations:int=1, enforceChirality:bool=True):

        self.name = name
        self.nr_of_conformations = nr_of_conformations
        assert(type(initial_state_mol) == Chem.Mol)
        assert(type(final_state_mol) == Chem.Mol)

        logger.info(f"Nr of conformations generated per tautomer: {nr_of_conformations}")
        logger.info(f"Chirality enforced: {enforceChirality}")

        self.initial_state_mol:Chem.Mol = initial_state_mol
        self.initial_state_mol_nx:nx.Graph = self._mol_to_nx(initial_state_mol)

        self.final_state_mol:Chem.Mol = final_state_mol
        self.final_state_mol_nx:nx.Graph = self._mol_to_nx(final_state_mol)

        initial_state_ani_input = self._from_mol_to_ani_input(self.initial_state_mol, enforceChirality)
        self.initial_state_ligand_atoms = initial_state_ani_input['ligand_atoms']
        self.initial_state_ligand_bonds = initial_state_ani_input['ligand_bonds']
        self.initial_state_ligand_coords = initial_state_ani_input['ligand_coords']
        self.initial_state_ligand_topology:md.Topology = initial_state_ani_input['ligand_topology']
        self.initial_state_ase_mol:Atoms = initial_state_ani_input['ase_mol']

        final_state_ani_input = self._from_mol_to_ani_input(self.final_state_mol, enforceChirality)
        self.final_state_ligand_atoms = final_state_ani_input['ligand_atoms']
        self.final_state_ligand_bonds = final_state_ani_input['ligand_bonds']
        self.final_state_ligand_coords = final_state_ani_input['ligand_coords']
        self.final_state_ligand_topology:md.Topology = final_state_ani_input['ligand_topology']
        self.final_state_ase_mol:Atoms = final_state_ani_input['ase_mol']

        # attributes for the protocol
        self.hybrid_hydrogen_idx_at_lambda_1:int = -1 # the dummy hydrogen
        self.hybrid_atoms:str = ''
        self.hybrid_ligand_idxs = []
        self.hybrid_coords:list = []
        self.hybrid_topology:md.Topology = md.Topology()       
        self.heavy_atom_hydrogen_donor_idx:int = -1 # the heavy atom that losses the hydrogen
        self.hydrogen_idx:int = -1 # the tautomer hydrogen
        self.hybrid_hydrogen_idx_at_lambda_0:int = -1
        self.heavy_atom_hydrogen_acceptor_idx:int = -1 # the heavy atom that accepts the hydrogen

        self.ligand_in_water_atoms:str = ""
        self.ligand_in_water_coordinates:list = []
        self.ligand_in_water_topology:md.Topology = md.Topology()

        # restraints for the ligand system
        self.ligand_restraints = []
        # restraint for the dummy - heavy atom bond
        self.hybrid_ligand_restraints = []
        # restraints for the solvent
        self.solvent_restraints = []
        # COM restraint
        self.com_restraints = []


    def _mol_to_nx(self, mol:Chem.Mol):
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                    atomic_num=atom.GetAtomicNum(),
                    formal_charge=atom.GetFormalCharge(),
                    chiral_tag=atom.GetChiralTag(),
                    hybridization=atom.GetHybridization(),
                    num_explicit_hs=atom.GetNumExplicitHs(),
                    is_aromatic=atom.GetIsAromatic())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    bond_type=bond.GetBondType())
        return G


    @property
    def initial_state_ligand_degeneracy(self):
        # Now iterate over graph isomorphisms
        from networkx.algorithms.isomorphism import GraphMatcher
        graph_matcher = GraphMatcher(self.initial_state_mol_nx, self.initial_state_mol_nx)
        degeneracy = sum([1 for isomorphism in graph_matcher.match()])
        return degeneracy

    @property
    def initial_state_entropy_correction(self):
        return (- kT * np.log(self.initial_state_ligand_degeneracy)).in_units_of(unit.kilocalorie_per_mole)

    @property
    def final_state_entropy_correction(self):
        return (- kT * np.log(self.final_state_ligand_degeneracy)).in_units_of(unit.kilocalorie_per_mole)

    @property
    def final_state_ligand_degeneracy(self):
        # Now iterate over graph isomorphisms
        from networkx.algorithms.isomorphism import GraphMatcher
        graph_matcher = GraphMatcher(self.final_state_mol_nx, self.final_state_mol_nx)
        degeneracy = sum([1 for isomorphism in graph_matcher.match()])
        return degeneracy


    def add_droplet(self, topology:md.Topology, 
                    coordinates:unit.quantity.Quantity, 
                    diameter:unit.quantity.Quantity=(30.0 * unit.angstrom),
                    restrain_hydrogens=True, 
                    file=None)->md.Trajectory:
        """
        Adding a droplet with a given diameter around a small molecule.
        
        Parameters
        ----------
        topology: md.Topology
            topology of the molecule
        coordinates: np.array, unit'd
        diameter: float, unit'd
        file: str
            if file is provided the final droplet pdb is either kept and can be reused or if 
            file already exists it will be used to create the same droplet.
        Returns
        ----------
        A mdtraj.Trajectory object with the ligand centered in the solvent for inspection. 
        """

        assert(type(diameter) == unit.Quantity)
        assert(type(topology) == md.Topology)
        assert(type(coordinates) == unit.Quantity)
        
        logger.info('Adding droplet ...')
        # get topology from mdtraj to PDBfixer via pdb file 
        radius = diameter.value_in_unit(unit.angstrom)/2
        center = np.array([radius, radius, radius])

        # if no solvated pdb file is provided generate one
        if file:
            # read in the file with the defined droplet
            pdb_filepath=file
        else:
            # generage a one time droplet
            pdb_filepath=f"tmp{random.randint(1,10000000)}.pdb"
        
        if not os.path.exists(pdb_filepath):

            # mdtraj works with nanomter
            md.Trajectory(coordinates.value_in_unit(unit.nanometer), topology).save_pdb(pdb_filepath)
            pdb = PDBFixer(filename=pdb_filepath)
            os.remove(pdb_filepath)

            # put the ligand in the center
            l_in_nanometer = diameter.value_in_unit(unit.nanometer)
            pdb.positions = np.array(pdb.positions.value_in_unit(unit.nanometer)) + (l_in_nanometer/2)
            # add water
            logger.info('Adding water ...')

            pdb.addSolvent(boxVectors=(Vec3(l_in_nanometer, 0.0, 0.0), Vec3(0.0, l_in_nanometer, 0.0), Vec3(0.0, 0.0, l_in_nanometer)))
            # get topology from PDBFixer to mdtraj # NOTE: a second tmpfile - not happy about this 
            from simtk.openmm.app import PDBFile
            PDBFile.writeFile(pdb.topology, pdb.positions, open(pdb_filepath, 'w'))
            # load pdb in parmed
            logger.info('Load with parmed ...')
            structure = pm.load_file(pdb_filepath)
            os.remove(pdb_filepath)

            # search for residues that are outside of the cutoff and delete them
            to_delete = []
            logger.info('Flag residues ...')

            for residue in structure.residues:
                for atom in residue:

                    p1 = np.array([atom.xx, atom.xy, atom.xz])
                    p2 = center

                    squared_dist = np.sum((p1-p2)**2, axis=0)
                    dist = np.sqrt(squared_dist)
                    if dist > radius+1: # NOTE: distance must be greater than radius + 1 Angstrom
                        to_delete.append(residue)
            
            logger.info('Delete residues ...')    
            for residue in list(set(to_delete)):
                logging.info('Remove: {}'.format(residue))
                structure.residues.remove(residue)
            
            structure.write_pdb(pdb_filepath)
        
        
        # load pdb with mdtraj
        traj = md.load(pdb_filepath)
        if not file:
            os.remove(pdb_filepath)

        # set coordinates #NOTE: note the xyz[0]
        self.ligand_in_water_coordinates = traj.xyz[0] * unit.nanometer

        # generate atom string
        atom_list = []
        for atom in traj.topology.atoms:
            atom_list.append(atom.element.symbol)
        
        # set atom string
        self.ligand_in_water_atoms = ''.join(atom_list)
        # set mdtraj topology
        self.ligand_in_water_topology = traj.topology

        # set FlattBottomRestraintToCenter restraints
        self.solvent_restraints = []
        for residue in traj.topology.residues:   
            if residue.is_water:
                for atom in residue.atoms:
                    if str(atom.element.symbol) == 'O':
                        self.solvent_restraints.append(
                            FlatBottomRestraintToCenter(sigma=0.1 * unit.angstrom, 
                                                        point=center * unit.angstrom, 
                                                        radius=(diameter/2),  
                                                        atom_idx = atom.index, 
                                                        active_at_lambda=-1))
                        print('Adding restraint to center to {}'.format(atom.index))
        
        if restrain_hydrogens:
            for residue in traj.topology.residues:   
                if residue.is_water:
                    oxygen_idx = None
                    hydrogen_idxs = []
                    for atom in residue.atoms:
                        if str(atom.element.symbol) == 'O':
                            oxygen_idx = atom.index
                        elif str(atom.element.symbol) == 'H':
                            hydrogen_idxs.append(atom.index)
                        else:
                            raise RuntimeError('Water should only consist of O and H atoms.')
                    self.solvent_restraints.append(FlatBottomRestraint(sigma=0.1 * unit.angstrom, atom_i_idx=oxygen_idx, atom_j_idx=hydrogen_idxs[0], atoms=self.ligand_in_water_atoms))
                    self.solvent_restraints.append(FlatBottomRestraint(sigma=0.1 * unit.angstrom, atom_i_idx=oxygen_idx, atom_j_idx=hydrogen_idxs[1], atoms=self.ligand_in_water_atoms))
            
        # return a mdtraj object for visual check
        return md.Trajectory(self.ligand_in_water_coordinates.value_in_unit(unit.nanometer), 
                                self.ligand_in_water_topology)

        
    def add_COM_for_hybrid_ligand(self, center):

        assert(type(center) == unit.Quantity)
        atoms = self.hybrid_atoms
        idx = self.hybrid_ligand_idxs
        self.add_COM_restraint(sigma=0.2 * unit.angstrom, point=center, atom_idx=idx, atoms=atoms)


    def add_COM_restraint(self, sigma:unit.Quantity, point, atom_idx:list, atoms:str):

        # add center of mass restraint
        com_restraint = CenterOfMassRestraint(sigma=sigma, point=point, atom_idx=atom_idx, atoms=atoms)
        self.com_restraints.append(com_restraint)


    def perform_tautomer_transformation_forward(self):
        """
        Performs a tautomer transformation from the initial state to the final state 
        and sets parameter and restraints using the indexing of the initial state mol.
        """

        self.ligand_restraints = []
        self.hybrid_ligand_restraints = []
        m1 = copy.deepcopy(self.initial_state_mol)
        m2 = copy.deepcopy(self.final_state_mol)
        self._perform_tautomer_transformation(m1, m2, self.initial_state_ligand_bonds)
        self._generate_hybrid_structure(self.initial_state_ligand_atoms, self.initial_state_ligand_coords[0], self.initial_state_ligand_topology)

    def perform_tautomer_transformation_reverse(self):
        """
        Performs a tautomer transformation from the final state to the initial state 
        and sets parameter and restraints using the indexing of the final state mol.
        """

        self.ligand_restraints = []
        self.hybrid_ligand_restraints = []
        m1 = copy.deepcopy(self.final_state_mol)
        m2 = copy.deepcopy(self.initial_state_mol)
        self._perform_tautomer_transformation(m1, m2, self.final_state_ligand_bonds)
        self._generate_hybrid_structure(self.final_state_ligand_atoms, self.final_state_ligand_coords[0], self.final_state_ligand_topology)

    def _from_mol_to_ani_input(self, mol:Chem.Mol, enforceChirality:bool):
        """
        Helper function - does not need to be called directly.
        Generates ANI input from a rdkit mol object
        """
        
        # generate atom list
        atom_list = []
        for a in mol.GetAtoms():
            atom_list.append(a.GetSymbol())

        if 'S' in atom_list:
            raise NotImplementedError('Sulfur not yet included in ANI.')
        # generate conformations
        mol = self._generate_conformations_from_mol(mol, self.nr_of_conformations, enforceChirality)
        # generate coord list
        coord_list = []

        # add conformations to coord_list
        for conf_idx in range(mol.GetNumConformers()):
            tmp_coord_list = []
            for a in mol.GetAtoms():
                pos = mol.GetConformer(conf_idx).GetAtomPosition(a.GetIdx())
                tmp_coord_list.append([pos.x, pos.y, pos.z])
            coord_list.append(np.array(tmp_coord_list) * unit.angstrom)

        # generate bond list of heavy atoms to hydrogens
        bond_list = []
        for b in mol.GetBonds():
            a1 = (b.GetBeginAtom())
            a2 = (b.GetEndAtom())

            if a1.GetSymbol() == 'H' or a2.GetSymbol() == 'H':
                bond_list.append((a1.GetIdx(), a2.GetIdx()))
    
        # get mdtraj topology
        n = random.randint(1,10000000)
        # TODO: use tmpfile for this https://stackabuse.com/the-python-tempfile-module/ or io.StringIO
        _ = write_pdb(mol, f"tmp{n}.pdb")
        topology = md.load(f"tmp{n}.pdb").topology
        os.remove(f"tmp{n}.pdb")
        
        ani_input =  {'ligand_atoms' : ''.join(atom_list), 
                'ligand_coords' : coord_list, 
                'ligand_topology' : topology,
                'ligand_bonds' : bond_list
                }

        # generate ONE ASE object if thermocorrections are needed
        ase_atom_list = []
        for e, c in zip(ani_input['ligand_atoms'], coord_list[0]):
                c_list = (c[0].value_in_unit(unit.angstrom), 
                        c[1].value_in_unit(unit.angstrom), 
                        c[2].value_in_unit(unit.angstrom)) 
                ase_atom_list.append(Atom(e, c_list))

        mol = Atoms(ase_atom_list)
        ani_input['ase_mol'] = mol
        return ani_input


    def _generate_conformations_from_mol(self, mol:Chem.Mol, nr_of_conformations:int, enforceChirality:bool):
        """
        Helper function - does not need to be called directly.
        Generates conformations from a rdkit mol object.        
        """  

        charge = 0
        for at in mol.GetAtoms():
            if at.GetFormalCharge() != 0:
                charge += int(at.GetFormalCharge())          

        if charge != 0:
            print(Chem.MolToSmiles(mol))
            raise NotImplementedError('Charged system')

        mol.SetProp("smiles", Chem.MolToSmiles(mol))
        mol.SetProp("charge", str(charge))
        mol.SetProp("name", str(self.name))

        # generate numConfs for the smiles string 
        new_confs = Chem.rdDistGeom.EmbedMultipleConfs(mol, 
                                            numConfs=nr_of_conformations, 
                                            enforceChirality=enforceChirality,
                                            ignoreSmoothingFailures=True) # NOTE enforceChirality!
        
        assert(int(mol.GetNumConformers()) != 0)
        assert(int(mol.GetNumConformers()) == nr_of_conformations)
        

        return mol


    def _perform_tautomer_transformation(self, m1:Chem.Mol, m2:Chem.Mol, ligand_bonds:list):
        """
        Helper function - does not need to be called directly.
        performs the actual tautomer transformation from m1 to m2.
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

        # adding ligand constraints for heavy atom - hydrogen
        for b in ligand_bonds:
            a1 =  b[0]
            a2 =  b[1]
            self.ligand_restraints.append(FlatBottomRestraint(sigma=0.1 * unit.angstrom, atom_i_idx=a1, atom_j_idx=a2, atoms=atoms))       

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
        
        # add NCMC restraints
        r1 = FlatBottomRestraint( sigma=0.1 * unit.angstrom, atom_i_idx=donor, atom_j_idx=hydrogen_idx_that_moves, atoms=atoms, active_at_lambda=1)
        r2 = FlatBottomRestraint( sigma=0.1 * unit.angstrom, atom_i_idx=acceptor, atom_j_idx=hydrogen_idx_that_moves, atoms=atoms, active_at_lambda=0)

        self.heavy_atom_hydrogen_donor_idx = donor
        self.hydrogen_idx = hydrogen_idx_that_moves
        self.hybrid_hydrogen_idx_at_lambda_0 = hydrogen_idx_that_moves
        self.heavy_atom_hydrogen_acceptor_idx = acceptor
        self.ncmc_restraints = [r1,r2]



    def _generate_hybrid_structure(self, ligand_atoms:str, ligand_coords, ligand_topology:md.Topology):
        """
        Helper function - does not need to be called directly.
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
        self.hybrid_ligand_idxs = [i for i in range(len(hybrid_atoms))]
        energy_function.use_pure_ani1ccx = True
        # generate MC mover to get new hydrogen position
        hydrogen_mover = MC_Mover(self.heavy_atom_hydrogen_donor_idx, 
                                self.hydrogen_idx, 
                                self.heavy_atom_hydrogen_acceptor_idx,
                                ligand_atoms)


        min_e = 100 * unit.kilocalorie_per_mole
        min_coordinates = None

        for _ in range(100):
            hybrid_coord = hydrogen_mover._move_hydrogen_to_acceptor_idx(ligand_coords, override=False)
            e, _, __, ___ = energy_function.calculate_energy(hybrid_coord, lambda_value=1.0)
            if e < min_e:
                min_e = e
                min_coordinates = hybrid_coord 
        
        self.hybrid_hydrogen_idx_at_lambda_1 = len(hybrid_atoms) -1
        self.hybrid_coords = min_coordinates

        # add restraint between dummy atom and heavy atom
        self.hybrid_ligand_restraints.append(FlatBottomRestraint(sigma=0.1 * unit.angstrom, atom_i_idx=self.hybrid_hydrogen_idx_at_lambda_1, atom_j_idx=self.heavy_atom_hydrogen_acceptor_idx, atoms=hybrid_atoms))

        # add to mdtraj ligand topology a new hydrogen
        hybrid_topology = copy.deepcopy(ligand_topology)
        dummy_atom = hybrid_topology.add_atom('H', md.element.hydrogen, hybrid_topology.residue(-1))
        hybrid_topology.add_bond(hybrid_topology.atom(self.heavy_atom_hydrogen_acceptor_idx), dummy_atom)
        self.hybrid_topology = hybrid_topology

    def generate_mining_minima_structures(self, rmsd_threshold:float=0.1, include_entropy_correction:bool=False)->(list, unit.Quantity, list):
        """
        Minimizes and filters conformations based on a RMSD threshold.
        Parameters
        ----------
        rmsd_threshol : float
            Treshold for RMSD filtering.
        include_entropy_correction : bool
            whether to include a degeneracy correction or not
        Returns
        -------
        confs_traj : list
            list of md.Trajectory objects with filtered conformations
        e : unit.Quantity
            free energy difference dG(final_state - initial_state)
        minimum_energies : list
            list of energies for the different minimum conformations
        """

        def prune_conformers(mol:Chem.Mol, energies:list, rmsd_threshold:float)->(Chem.Mol, list):
            """
            Adopted from: https://github.com/skearnes/rdkit-utils/blob/master/rdkit_utils/conformers.py
            Prune conformers from a molecule using an RMSD threshold, starting
            with the lowest energy conformer.
            Parameters
            ----------
            mol : RDKit Mol
                Molecule.
            Returns
            -------
            A new RDKit Mol containing the chosen conformers, sorted by
            increasing energy.
            """
            rmsd = get_conformer_rmsd(mol)
            sort = np.argsort([x.value_in_unit(unit.kilocalorie_per_mole) for x in energies])  # sort by increasing energy
            keep = []  # always keep lowest-energy conformer
            discard = []
            for i in sort:

                # always keep lowest-energy conformer
                if len(keep) == 0:
                    keep.append(i)
                    continue

                # get RMSD to selected conformers
                this_rmsd = rmsd[i][np.asarray(keep, dtype=int)]

                # discard conformers within the RMSD threshold
                if np.all(this_rmsd >= rmsd_threshold):
                    keep.append(i)
                else:
                    discard.append(i)

            # create a new molecule to hold the chosen conformers
            # this ensures proper conformer IDs and energy-based ordering
            new_mol = Chem.Mol(mol)
            new_mol.RemoveAllConformers()
            conf_ids = [conf.GetId() for conf in mol.GetConformers()]
            filtered_energies = []
            for i in keep:
                conf = mol.GetConformer(conf_ids[i])
                filtered_energies.append(energies[i])
                new_mol.AddConformer(conf, assignId=True)
            return new_mol, filtered_energies


        def get_conformer_rmsd(mol)->list:
            """
            Calculate conformer-conformer RMSD.
            Parameters
            ----------
            mol : RDKit Mol
                Molecule.
            """
            rmsd = np.zeros((mol.GetNumConformers(), mol.GetNumConformers()),
                            dtype=float)


            #mol = _remove_hydrogens(copy.deepcopy(mol))
            for i, ref_conf in enumerate(mol.GetConformers()):
                for j, fit_conf in enumerate(mol.GetConformers()):
                    if i >= j:
                        continue
                    rmsd[i, j] = AllChem.GetBestRMS(mol, mol, ref_conf.GetId(),
                                                    fit_conf.GetId())
                    rmsd[j, i] = rmsd[i, j]
            return rmsd


        def calculate_weighted_energy(e_list):
            #G = -RT ln Σ exp(-G/RT)

            l = []
            for energy in e_list:
                v = ((-1) * (energy)) / (gas_constant * temperature)
                l.append(v)
        
            e_bw = (-1) * gas_constant * temperature * (logsumexp(l)) 
            return e_bw


        bw_energies = []
        confs_traj = []
        minimum_energies = []

        for ase_mol, rdkit_mol, ligand_atoms, ligand_coords, top, entropy_correction in zip(
        [self.initial_state_ase_mol, self.final_state_ase_mol], 
        [copy.deepcopy(self.initial_state_mol), copy.deepcopy(self.final_state_mol)],
        [self.initial_state_ligand_atoms, self.final_state_ligand_atoms], 
        [self.initial_state_ligand_coords, self.final_state_ligand_coords], 
        [self.initial_state_ligand_topology, self.final_state_ligand_topology],
        [self.initial_state_entropy_correction, self.final_state_entropy_correction]): 

            print('Mining Minima starting ...')
            model = torchani.models.ANI1ccx()
            model = model.to(device)

            energy_function = ANI1_force_and_energy(
                                                    model = model,
                                                    atoms = ligand_atoms,
                                                    mol = ase_mol,
                                                    use_pure_ani1ccx = True
                                                )
            traj = []
            energies = []
            for n_conf, coords in enumerate(ligand_coords):
                # minimize
                print(f"Conf: {n_conf}")
                minimized_coords, _ = energy_function.minimize(coords)
                single_point_energy, _, __ = energy_function.calculate_energy(minimized_coords)
                try:
                    thermochemistry_correction = energy_function.get_thermo_correction(minimized_coords)  
                except ValueError:
                    logger.critical('Imaginary frequencies present - found transition state.')
                    continue

                if include_entropy_correction:
                    energies.append(single_point_energy + thermochemistry_correction + entropy_correction)
                else:
                    energies.append(single_point_energy + thermochemistry_correction)

                # update the coordinates in the rdkit mol
                for atom in rdkit_mol.GetAtoms():
                    conf = rdkit_mol.GetConformer(n_conf)
                    new_coords = Geometry.rdGeometry.Point3D()
                    new_coords.x = (minimized_coords[atom.GetIdx()][0]).value_in_unit(unit.angstrom)
                    new_coords.y = minimized_coords[atom.GetIdx()][1].value_in_unit(unit.angstrom)
                    new_coords.z = minimized_coords[atom.GetIdx()][2].value_in_unit(unit.angstrom)
                    conf.SetAtomPosition(atom.GetIdx(), new_coords)
   
            # aligne the molecules
            AllChem.AlignMolConformers(rdkit_mol)
            min_and_filtered_rdkit_mol, filtered_energies = prune_conformers(rdkit_mol, copy.deepcopy(energies), rmsd_threshold=rmsd_threshold)

            # generate mdtraj object
            traj = []
            for conf_idx in range(min_and_filtered_rdkit_mol.GetNumConformers()):
                tmp_coord_list = []
                for a in min_and_filtered_rdkit_mol.GetAtoms():
                    pos = min_and_filtered_rdkit_mol.GetConformer(conf_idx).GetAtomPosition(a.GetIdx())
                    tmp_coord_list.append([pos.x, pos.y, pos.z])
                tmp_coord_list = np.array(tmp_coord_list) * unit.angstrom
                traj.append(tmp_coord_list.value_in_unit(unit.nanometer))

            confs_traj.append(md.Trajectory(traj, top))
            minimum_energies.append(filtered_energies)
            bw_energies.append(calculate_weighted_energy(filtered_energies))
            print('Mining Minima finished ...')

        e = (bw_energies[1] - bw_energies[0])
        return confs_traj, e, minimum_energies
