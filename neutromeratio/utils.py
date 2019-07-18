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
from scipy.stats import norm

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

def reduce(E):
    """
    Convert unit'd energy into a unitless reduced potential energy.

    In NVT:
        u(x) = U(x) / kBT
    """
    return E / kT

def energy_function(coordinates, model, species, device):

    _, energy_in_hartree = model((species, torch.tensor([coordinates],
                           requires_grad=True, device=device, dtype=torch.float32)))
    # convert energy from hartrees to kJ/mol
    return (energy_in_hartree.item()* hartree_to_kJ_mol) * energy_unit


class MC_mover(object):

    def __init__(self, donor_idx, hydrogen_idx, acceptor_idx, atom_list):
        self.donor_idx = donor_idx
        self.hydrogen_idx = hydrogen_idx
        self.acceptor_idx = acceptor_idx
        self.atom_list = atom_list
        self.mc_accept_counter = 0
        self.mc_reject_counter = 0
        self.bond_lenght_dict = { 'CH' : 1.09 * unit.angstrom,
                                'OH' : 0.96 * unit.angstrom,
                                'NH' : 1.01 * unit.angstrom
                                }
        # a multiplicator to the equilibrium bond length to get the mean bond length
        self.acceptor_mod_bond_length = 1.0
        self.donor_mod_bond_length = 1.0
        # element of the hydrogen acceptor and donor
        self.acceptor_element = self.atom_list[self.acceptor_idx]
        self.donor_element = self.atom_list[self.donor_idx]
        # the equilibrium bond length taken from self.bond_length_dict
        self.acceptor_hydrogen_equilibrium_bond_length = self.bond_lenght_dict['{}H'.format(self.acceptor_element)]
        self.donor_hydrogen_equilibrium_bond_length = self.bond_lenght_dict['{}H'.format(self.donor_element)]
        # the stddev for the bond length
        self.acceptor_hydrogen_stddev_bond_length = 0.15 * unit.angstrom
        self.donor_hydrogen_stddev_bond_length = 0.15 * unit.angstrom
        # the mean bond length is the bond length that is actually used for proposing coordinates
        @property
        def acceptor_hydrogen_mean_bond_length(self):
            return self.acceptor_hydrogen_equilibrium_bond_length * self.acceptor_mod_bond_length

        @property
        def donor_hydrogen_mean_bond_length(self):
            return self.donor_hydrogen_equilibrium_bond_length * self.donor_mod_bond_length

    def perform_mc_move(self, coordinates, model, species, device):
        """
        Moves a hydrogen (self.hydrogen_idx) from a starting position connected to a heavy atom
        donor (self.donor_idx) to a new position around an acceptor atom (self.acceptor_idx).
        The new coordinates are sampled from a radial distribution, with the center beeing the acceptor atom,
        the mean: mean_bond_length = self.acceptor_hydrogen_equilibrium_bond_length * self.acceptor_mod_bond_length
        and standard deviation: self.acceptor_hydrogen_stddev_bond_length.
        Calculates the log probability of the forward and reverse proposal and returns the work.
        """
        # convert coordinates to angstroms
        # coordinates_before_move
        coordinates_A = coordinates.value_in_unit(unit.angstrom)
        #coordinates_after_move
        coordinates_B = self._move_hydrogen_to_acceptor_idx(copy.deepcopy(coordinates_A))
        delta_u = reduce(energy_function(coordinates_B, model, species, device) - energy_function(coordinates_A, model, species, device))

        # log probability of forward proposal from the initial coordinates (coordinate_A) to proposed coordinates (coordinate_B)
        log_p_forward = self.log_probability_of_proposal_to_B(coordinates_A, coordinates_B)
        # log probability of reverse proposal given the proposed coordinates (coordinate_B)
        log_p_reverse = self.log_probability_of_proposal_to_A(coordinates_B, coordinates_A)
        work = - ((- delta_u) + (log_p_reverse - log_p_forward))
        return coordinates_B * unit.angstrom, work

    def _log_probability_of_radial_proposal(self, r, r_mean, r_stddev):
        """Logpdf of N(r : mu=r_mean, sigma=r_stddev)"""
        return norm.logpdf(r, loc=r_mean, scale=r_stddev)

    def log_probability_of_proposal_to_B(self, X, X_prime):
        """log g_{A \to B}(X --> X_prime)"""
        # test if acceptor atoms are similar in both coordinate sets
        assert(np.allclose(X[self.acceptor_idx], X_prime[self.acceptor_idx]))
        # calculates the effective bond length of the proposed conformation between the
        # hydrogen atom and the acceptor atom
        r = np.linalg.norm(X_prime[self.hydrogen_idx] - X_prime[self.acceptor_idx])
        return self._log_probability_of_radial_proposal(r, self.acceptor_hydrogen_mean_bond_length * (1/unit.angstrom), self.acceptor_hydrogen_stddev_bond_length * (1/unit.angstrom))

    def log_probability_of_proposal_to_A(self, X, X_prime):
        """log g_{B \to A}(X --> X_prime)"""
        assert(np.allclose(X[self.donor_idx], X_prime[self.donor_idx]))
        # calculates the effective bond length of the initial conformation between the
        # hydrogen atom and the donor atom
        r = np.linalg.norm(X_prime[self.hydrogen_idx] - X_prime[self.donor_idx])
        return self._log_probability_of_radial_proposal(r, self.donor_hydrogen_mean_bond_length * (1/unit.angstrom), self.donor_hydrogen_stddev_bond_length * (1/unit.angstrom))


    def accept_reject(self, log_P_accept: float) -> bool:
        """Perform acceptance/rejection check according to the Metropolis-Hastings acceptance criterium."""
        # think more about symmetric
        return (log_P_accept > 0.0) or (random.random() < math.exp(log_P_accept))

    def _move_hydrogen_to_acceptor_idx(self, coordinates_in_angstroms):
        """Moves a single hydrogen (specified in self.hydrogen_idx) from a donor
        atom (self.donor_idx) to a new position around the acceptor atom (self.acceptor_idx).
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
            # sample a random direction
            unit_vector = np.random.randn(ndim)
            unit_vector /= np.linalg.norm(unit_vector, axis=0)
            # sample a random length
            effective_bond_length = (np.random.randn() * self.acceptor_hydrogen_stddev_bond_length + self.acceptor_hydrogen_mean_bond_length)
            return (unit_vector * effective_bond_length)


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
