from openmmtools.constants import kB
from simtk import unit
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
import torchani
import torch
import random
import copy
import math
from tqdm import tqdm

# temperature, mass, and related constants
temperature = 300 * unit.kelvin
kT = kB * temperature

# openmm units
mass_unit = unit.dalton
distance_unit = unit.nanometer
time_unit = unit.femtosecond
energy_unit = unit.kilojoule_per_mole
speed_unit = distance_unit / time_unit
force_unit = unit.kilojoule_per_mole / unit.nanometer

# ANI-1 units and conversion factors
ani_distance_unit = unit.angstrom
hartree_to_kJ_mol = 2625.499638
ani_energy_unit = hartree_to_kJ_mol * unit.kilojoule_per_mole # simtk.unit doesn't have hartree?
nm_to_angstroms = (1.0 * distance_unit) / (1.0 * ani_distance_unit)

def get_donor_atom_idx(m1, m2):
    m1 = copy.deepcopy(m1)
    m2 = copy.deepcopy(m2)
    # find substructure and generate mol from substructure
    sub = rdFMCS.FindMCS([m1, m2], bondCompare=Chem.rdFMCS.BondCompare.CompareOrder.CompareAny)
    mcsp = Chem.MolFromSmarts(sub.smartsString, False)
    g = Chem.MolFromSmiles(Chem.MolToSmiles(mcsp, allHsExplicit=True), sanitize=False)
    substructure_idx_m1 = m1.GetSubstructMatch(g)

    #get idx of hydrogen that moves to new position
    hydrogen_idx_that_moves = -1
    for a in m1.GetAtoms():
        if a.GetIdx() not in substructure_idx_m1:
            print('m1: Index of atom that moves: {}.'.format(a.GetIdx()))
            hydrogen_idx_that_moves = a.GetIdx()
    AllChem.Compute2DCoords(m1)
    display(mol_with_atom_index(m1))

    # get idx of connected heavy atom which is the donor atom
    # there can only be one neighbor, therefor it is valid to take the first neighbor of the hydrogen
    donor = int(m1.GetAtomWithIdx(hydrogen_idx_that_moves).GetNeighbors()[0].GetIdx())
    return { 'donor': donor, 'hydrogen_idx' : hydrogen_idx_that_moves }


def write_xyz_traj_file(atom_list, coordinates, name='test'):

    if os.path.exists('traj_{}.xyz'.format(name)):
        f = open('traj_{}.xyz'.format(name), 'a')
    else:
        f = open('traj_{}.xyz'.format(name), 'w')

    for frame in coordinates:
        frame_in_angstrom = frame.value_in_unit(unit.angstrom)
        f.write('{}\n'.format(len(atom_list)))
        f.write('{}\n'.format('writing mols'))
        for atom, coordinate in zip(atom_list, frame_in_angstrom):
            f.write('  {:2}   {: 11.9f}  {: 11.9f}  {: 11.9f}\n'.format(atom, coordinate[0], coordinate[1], coordinate[2]))

def write_xyz_file(atom_list, coordinates, name='test', identifier='0_0'):

    f = open('mc_confs/{}_{}.xyz'.format(name, identifier), 'w')
    f.write('{}\n'.format(len(atom_list)))
    f.write('{}\n'.format('writing mols'))
    coordinates_in_angstroms = coordinates.value_in_unit(unit.angstrom)
    for atom, coordinate in zip(atom_list, coordinates_in_angstroms):
        f.write('  {:2}   {: 11.9f}  {: 11.9f}  {: 11.9f}\n'.format(atom, coordinate[0], coordinate[1], coordinate[2]))

def write_pdb(mol, name, tautomer_id):
    # write a pdf file using the name and tautomer_id
    Chem.MolToPDBFile(mol, '../data/md_sampling/{}/{}_{}.pdb'.format(name, name, tautomer_id))
    return Chem.MolToPDBBlock(mol)

def generate_rdkit_mol(smiles):
    # geneartes a rdkit mol object with 3D coordinates from smiles
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    display(mol_with_atom_index(m))
    AllChem.EmbedMolecule(m)
    return m

def from_mol_to_ani_input(mol):
    # generates atom_list and coord_list entries from rdkit mol
    atom_list = []
    coord_list = []
    for a in mol.GetAtoms():
        atom_list.append(a.GetSymbol())
        pos = mol.GetConformer().GetAtomPosition(a.GetIdx())
        coord_list.append([pos.x, pos.y, pos.z])
    return { 'atom_list' : ''.join(atom_list), 'coord_list' : coord_list}

def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol


def langevin(device,
             model,
             atom_list,
             x0,
             force,
             n_steps=100,
             stepsize=1 * unit.femtosecond,
             collision_rate=10/unit.picoseconds,
             temperature=300 * unit.kelvin,
             platform = 'cpu',
             progress_bar=False
            ):
    """Unadjusted Langevin dynamics.

    Parameters
    ----------
    device,
    model,
    atom_list
    x0 : array of floats, unit'd (distance unit)
        initial configuration
    force : callable, accepts a unit'd array and returns a unit'd array
        assumes input is in units of distance
        output is in units of energy / distance
    n_steps : integer
        number of Langevin steps
    stepsize : float > 0, in units of time
        finite timestep parameter
    collision_rate : float > 0, in units of 1/time
        controls the rate of interaction with the heat bath
    temperature : float > 0, in units of temperature
    progress_bar : bool
        use tqdm to show progress bar

    Returns
    -------
    traj : [n_steps + 1 x dim] array of floats, unit'd
        trajectory of samples generated by Langevin dynamics

    """

    assert(type(x0) == unit.Quantity)
    assert(type(stepsize) == unit.Quantity)
    assert(type(collision_rate) == unit.Quantity)
    assert(type(temperature) == unit.Quantity)

    # generate mass arrays
    mass_dict_in_daltons = {'H': 1.0, 'C': 12.0, 'N': 14.0, 'O': 16.0}
    masses = np.array([mass_dict_in_daltons[a] for a in atom_list]) * unit.daltons
    sigma_v = np.array([unit.sqrt(kB * temperature / m) / speed_unit for m in masses]) * speed_unit
    v0 = np.random.randn(len(sigma_v),3) * sigma_v[:,None]
    # convert initial state numpy arrays with correct attached units
    x = np.array(x0.value_in_unit(distance_unit)) * distance_unit
    v = np.array(v0.value_in_unit(speed_unit)) * speed_unit

    # traj is accumulated as a list of arrays with attached units
    traj = [x]
    energies = []
    # dimensionless scalars
    a = np.exp(- collision_rate * stepsize)
    b = np.sqrt(1 - np.exp(-2 * collision_rate * stepsize))

    species = model.species_to_tensor(atom_list).to(device).unsqueeze(0)
    # compute force on initial configuration
    F, e = force(x, device, model, species, platform)
    energies.append(e)
    trange = range(n_steps)
    if progress_bar:
        trange = tqdm(trange)
    for _ in trange:

        # v
        v += (stepsize * 0.5) * F / masses[:,None]
        # r
        x += (stepsize * 0.5) * v
        # o
        v = (a * v) + (b * sigma_v[:,None] * np.random.randn(*x.shape))
        # r
        x += (stepsize * 0.5) * v
        F, e = force(x, device, model, species, platform)
        # v
        v += (stepsize * 0.5) * F / masses[:,None]

        norm_F = np.linalg.norm(F)
        # report gradient norm
        if progress_bar:
            trange.set_postfix({'|force|': norm_F})
        energies.append(e)
        # check positions and forces are finite
        if (not np.isfinite(x).all()) or (not np.isfinite(norm_F)):
            print("Numerical instability encountered!")
            return traj
        traj.append(x)
    return traj, x, energies

def ANI1ccx_force_and_energy(x, device, model, species, platform):
    """
    Parameters
    ----------
    x : numpy array, unit'd
        coordinates

    Returns
    -------
    F : numpy array, unit'd
        force, with units of kJ/mol/nm attached
    E : energy in kJ/mol
    """
    assert(type(x) == unit.Quantity)
    x_in_nm = x.value_in_unit(unit.nanometer)

    coordinates = torch.tensor([x_in_nm],
                               requires_grad=True, device=device, dtype=torch.float32)

    # convert from nm to angstroms
    coordinates_in_angstroms = coordinates * nm_to_angstroms
    _, energy_in_hartree = model((species, coordinates_in_angstroms))

    # convert energy from hartrees to kJ/mol
    energy_in_kJ_mol = energy_in_hartree * hartree_to_kJ_mol

    # derivative of E (in kJ/mol) w.r.t. coordinates (in nm)
    derivative = torch.autograd.grad((energy_in_kJ_mol).sum(), coordinates)[0]

    if platform == 'cpu':
        F_in_openmm_unit = - np.array(derivative)[0]
    elif platform == 'cuda':
        F_in_openmm_unit = - np.array(derivative.cpu())[0]
    else:
        raise RuntimeError('Platform needs to be specified. Either CPU or CUDA.')

    return F_in_openmm_unit * (unit.kilojoule_per_mole / unit.nanometer), energy_in_kJ_mol.item()



class MC_mover(object):

    def __init__(self, donor_idx, hydrogen_idx, acceptor_idx, atom_list):
        self.donor_idx = donor_idx
        self.hydrogen_idx = hydrogen_idx
        self.acceptor_idx = acceptor_idx
        self.atom_list = atom_list
        self.accept_counter = 0
        self.reject_counter = 0
        self.bond_lenght_dict = { 'CH' : 1.09 * unit.angstrom,
                                    'OH' : 0.96 * unit.angstrom,
                                        'NH' : 1.01 * unit.angstrom}
        self.mod_bond_length = 1.0
        self.equilibrium_bond_length = self.bond_lenght_dict['{}H'.format(self.atom_list[self.acceptor_idx])]
        self.std_bond_length = 0.15

    def perform_mc_move(self, coordinates, ts, model, species, device):

        # convert coordinates to angstroms
        coordinates_before_move = coordinates.value_in_unit(unit.angstrom)

        # get energy befor MC move
        _, energy_in_hartree = model((species, torch.tensor([coordinates_before_move],
                               requires_grad=True, device=device, dtype=torch.float32)))
        # convert energy from hartrees to kJ/mol
        e_start = (energy_in_hartree.item()* hartree_to_kJ_mol) * energy_unit
        log_P_initial = self.compute_log_probability(e_start)

        coordinates_after_move = self._move_hydrogen_to_donor_idx(coordinates_before_move)

        # get energy after MC move
        _, energy_in_hartree = model((species, torch.tensor([coordinates_after_move],
                               requires_grad=True, device=device, dtype=torch.float32)))

        # convert energy from hartrees to kJ/mol
        e_finish = (energy_in_hartree.item()* hartree_to_kJ_mol) * energy_unit
        log_P_final = self.compute_log_probability(e_finish)
        work = -(log_P_final - log_P_initial)

        accept = self.accept_reject(e)
        coordinates_after_move = (coordinates_after_move* unit.angstrom)
        coordinates_before_move = (coordinates_before_move* unit.angstrom)

        if accept:
            self.accept_counter += 1
            return True, coordinates_after_move, e
        else:
            self.reject_counter += 1
            return False, coordinates_before_move, e

    def accept_reject(self, log_P_accept: float) -> bool:
        """Perform acceptance/rejection check according to the Metropolis-Hastings acceptance criterium."""
        # think more about symmetric
        return (log_P_accept > 0.0) or (random.random() < math.exp(log_P_accept))

    def compute_log_probability(self, total_energy_kJ_mol):
        """
        Compute log probability
        """
        beta = 1.0 / kT  # inverse temperature
        a = (-beta * total_energy_kJ_mol)
        print(a)
        prop_dist = np.random.randn() * self.std_bond_length + (self.equilibrium_bond_length / unit.angstrom)
        print(prop_dist)
        print(a*prop_dist)

        return a * prop_dist


    def _move_hydrogen_out_of_mol_env(self, coordinates_in_angstroms):
        """Moves a single hydrogen (specified in self.hydrogen_idx) from an acceptor
        atom (self.acceptor_idx) to a new position 10 Angstrom away from its current position.

        Parameters
        ----------
        coordinates_in_angstroms :numpy array, unit'd
            coordinates

        Returns
        -------
        coordinates_in_angstroms :numpy array, unit'd
            coordinates
        """

        # get coordinates of acceptor atom and hydrogen that moves
        hydrogen_coordinate = coordinates_in_angstroms[self.hydrogen_idx] * unit.angstrom
        # generate new hydrogen atom position and replace old coordinates
        coordinates_in_angstroms[self.hydrogen_idx] = hydrogen_coordinate * 10
        return coordinates_in_angstroms



    def _move_hydrogen_to_donor_idx(self, coordinates_in_angstroms):
        """Moves a single hydrogen (specified in self.hydrogen_idx) from an acceptor
        atom (self.acceptor_idx) to a new position around the donor atom (self.donor_idx).
        Parameters
        ----------
        coordinates_in_angstroms :numpy array, unit'd
            coordinates

        Returns
        -------
        coordinates_in_angstroms :numpy array, unit'd
            coordinates
        """
        def sample_spherical(ndim=3):
            """
            Generates new coordinates for a hydrogen around a heavy atom acceptor.
            Bond length is defined by the hydrogen - acceptor element equilibrium bond length,
            definded in self.bond_lenght_dict. Standard deviation of bond length is defined
            in self.std_bond_length.
            A bias to the bond length can be intruduced through self.mod_bond_length.
            The effective bond length is mean_bond_length *= self.mod_bond_length.
            """
            vec = np.random.randn(ndim)
            vec /= np.linalg.norm(vec, axis=0)
            mean_bond_length = self.equilibrium_bond_length * self.mod_bond_length
            bond_length = np.random.randn() * self.std_bond_length + (mean_bond_length / unit.angstrom)
            #print('Effective bond length: {}'.format(bond_length))
            #print('Equilibrium bond length: {}'.format(mean_bond_length))
            #print('Std bond length: {}'.format(std_bond_length))
            return vec * bond_length * unit.angstrom


        # get coordinates of acceptor atom and hydrogen that moves
        hydrogen_coordinate = coordinates_in_angstroms[self.hydrogen_idx] * unit.angstrom
        acceptor_coordinate = coordinates_in_angstroms[self.acceptor_idx] * unit.angstrom

        # generate new hydrogen atom position and replace old coordinates
        new_hydrogen_coordinate = acceptor_coordinate + sample_spherical()
        coordinates_in_angstroms[self.hydrogen_idx] = new_hydrogen_coordinate

        return coordinates_in_angstroms




def generate_nglview_object(top_file, traj_file):
    """
    Generates nglview object from topology and trajectory files.
    Parameters
    ----------
    top_file : file path to mdtraj readable topology file
    traj_file : file path to mdtraj readable trajectory file
    Returns
    -------
    view: nglview object
    """

    topology = md.load(top_file).topology
    ani_traj = md.load(traj_file, top=topology)

    view = nglview.show_mdtraj(ani_traj)
    return view
