import logging

import numpy as np
import torch
from openmmtools.constants import kB
from scipy.stats import norm
from simtk import unit
from torch.distributions.normal import Normal

from .constants import (bond_length_dict, device, mass_dict_in_daltons,
                        nm_to_angstroms, temperature, water_hoh_angle,
                        conversion_factor_radian_to_degree)

logger = logging.getLogger(__name__)


class BaseDistanceRestraint(object):

    def __init__(self, sigma: unit.Quantity, active_at: int):
        """
        Defines a distance restraint base class
        Parameters
        ----------
        sigma : in angstrom
        active_at : int
            Integer to indicccate at which state the restraint is fully active. Either 0 (for 
            lambda 0), or 1 (for lambda 1) or -1 (always active)
        """

        assert(type(sigma) == unit.Quantity)
        k = (kB * temperature) / (sigma**2)
        self.device = device
        assert(active_at ==1 or active_at == 0 or active_at ==-1 or active_at == -2)
        self.active_at = active_at
        self.k = torch.tensor(k.value_in_unit((unit.kilo * unit.joule) / ((unit.angstrom ** 2) *
                                                                          unit.mole)), dtype=torch.double, device=self.device, requires_grad=True)


class BaseAngleRestraint(object):

    def __init__(self, sigma: unit.Quantity, active_at: int):
        """
        Defines an angle restraint base class.
        Force constant taken from here:
        'Typically for bond angle A–B–C, if A and C are both hydrogen atoms, the force 
        constant is roughly 30 –35 kcal/mol*rad**2.' Development and Testing of a General Amber Force Field.
        Parameters (http://ambermd.org/antechamber/gaff.pdf)
        ----------
        sigma : in angstrom
        active_at : int
            Integer to indicate at which state the restraint is fully active. Either 0 (for 
            lambda 0), or 1 (for lambda 1) or -1 (always active)
        """

        assert(type(sigma) == unit.Quantity)
        k = (kB * temperature) / (sigma**2)  # k = 34.2159 kcal/mol*rad**2
        self.device = device
        assert(active_at ==1 or active_at == 0 or active_at ==-1 or active_at == -2)
        self.active_at = active_at
        self.k = torch.tensor(k.value_in_unit((unit.kilo * unit.joule) / ((unit.radian ** 2) * unit.mole)),
                              dtype=torch.double,
                              device=self.device,
                              requires_grad=True)
        print(self.k)



class BaseTorsionRestraint(object):

    def __init__(self, sigma: unit.Quantity, torsion_angle: unit.Quantity, active_at: int):
        """
        Defines an torsion restraint base class.
        ----------
        sigma : in angstrom
        active_at : int
            Integer to indicate at which state the restraint is fully active. Either 0 (for 
            lambda 0), or 1 (for lambda 1) or -1 (always active)
        """

        assert(type(sigma) == unit.Quantity)
        assert(type(torsion_angle) == unit.Quantity)

        k = (kB * temperature) / (sigma**2) # 
        self.device = device
        assert(active_at ==1 or active_at == 0 or active_at ==-1 or active_at == -2)
        self.active_at = active_at
        self.target_torsion_angle_in_degree = torsion_angle.value_in_unit(unit.degree)
        self.k = torch.tensor(k.value_in_unit((unit.kilo * unit.joule) / ((unit.degree ** 2) * unit.mole)),
                              dtype=torch.double,
                              device=self.device,
                              requires_grad=True)




class TorsionHarmonicRestraint(BaseTorsionRestraint):

    def __init__(self, sigma: unit.Quantity, torsion_angle: unit.Quantity, atom_idx :list, active_at: int=-1):

        assert (0 <= active_at <= 1 or active_at == -1 or active_at == -2)
        super().__init__(sigma, torsion_angle, active_at)

        self.device = device
        self.atom_idx = atom_idx
        self.active_at = active_at
        self.atom_i = atom_idx[0]
        self.atom_j = atom_idx[1]
        self.atom_k = atom_idx[2]
        self.atom_l = atom_idx[3]
        
    def _calculate_torsion(self, x):
        # calculating torsion -- taken from here: https://github.com/mdtraj/mdtraj/blob/master/mdtraj/geometry/dihedral.py


        dxij = (x[0][self.atom_j] - x[0][self.atom_i])
        dxjk = (x[0][self.atom_k] - x[0][self.atom_j])
        dxkl = (x[0][self.atom_l] - x[0][self.atom_k])

        c1 = torch.cross(-dxjk, -dxkl)
        c2 = torch.cross(dxij, -dxjk)

        p1 = torch.sum(dxij * c1)
        p1 *= torch.norm(dxjk)
        p2 = torch.sum(c1 * c2)

        torsion = torch.atan2(p1,p2)
        return torsion * conversion_factor_radian_to_degree


    def restraint(self, x):
        """       
        
        Parameters
        -------
        x : torch.Tensor
            coordinates
        Returns
        -------
        e : torch.Tensor
        """

        try:
            assert (type(x) == torch.Tensor)
        except AssertionError:
            assert (type(x) == unit.Quantity)
            x = torch.tensor([x.value_in_unit(unit.nanometer)],
                            requires_grad=True, device=self.device, dtype=torch.float32)
            

        torsion_in_degree = self._calculate_torsion(x)
        e = (0.5 * self.k) * (self.target_torsion_angle_in_degree - torsion_in_degree)**2
        
        logging.debug('Torsion harmonic restraint restraint_bias introduced: {:0.4f}'.format(e.item()))
        return e.to(device=self.device)



class PointAtomRestraint(BaseDistanceRestraint):

    def __init__(self, sigma: unit.Quantity, point: np.array, active_at: int):
        """
        Defines a Point to Atom restraint base class. 

        Parameters
        ----------
        sigma : in angstrom
        point : np.array 3D, value in angstrom
        active_at : int
            Integer to indicccate at which state the restraint is fully active. Either 0 (for 
            lambda 0), or 1 (for lambda 1) or -1 (always active)
        Returns
        -------
        e : float
            restraint_bias
        """

        super().__init__(sigma, active_at)
        assert(type(point) == np.ndarray)
        self.point = torch.tensor(point,
                                  dtype=torch.double,
                                  device=self.device,
                                  requires_grad=True)


class BondRestraint(BaseDistanceRestraint):

    def __init__(self, sigma: unit.Quantity,
                 atom_i_idx: int,
                 atom_j_idx: int,
                 atoms: str,
                 active_at: int = -1):
        """
        Defines a Atom to Atom restraint base class. 

        Parameters
        ----------
        sigma : in angstrom
        atom_i_idx : int
            Atom i to restraint
        atom_j_idx : int
            Atom j to restraint
        atoms: str
            Str of atoms to retrieve element information
        active_at : int
            Integer to indicccate at which state the restraint is fully active. Either 0 (for 
            lambda 0), or 1 (for lambda 1) or -1 (always active)
        """
        super().__init__(sigma, active_at)
        self.atom_i_element = atoms[atom_i_idx]
        self.atom_j_element = atoms[atom_j_idx]
        self.atom_i_idx = atom_i_idx
        self.atom_j_idx = atom_j_idx

        # get mean bond length
        try:
            self.mean_bond_length = (bond_length_dict[frozenset(
                [self.atom_i_element, self.atom_j_element])]).value_in_unit(unit.angstrom)
            self.upper_bound = self.mean_bond_length + 0.4
            self.lower_bound = self.mean_bond_length - 0.4

        except KeyError:
            logger.warning('Bond between: {} - {}'.format(self.atom_i_element, self.atom_j_element))
            logger.warning('Falling back to 1.5 Angstrom.')
            self.mean_bond_length = 1.5
            self.upper_bound = self.mean_bond_length + 0.2
            self.lower_bound = self.mean_bond_length - 0.2


class AngleHarmonicRestraint(BaseAngleRestraint):

    def __init__(self,
                 sigma: unit.Quantity,
                 atom_i_idx: int,
                 atom_j_idx: int,
                 atom_k_idx: int,
                 active_at: int = -1):
        """
        Restraints the angle between bond i-j and bond k-j
        """
        super().__init__(sigma, active_at)
        self.atom_i_idx = atom_i_idx
        self.atom_j_idx = atom_j_idx
        self.atom_k_idx = atom_k_idx
        self.water_angle = water_hoh_angle.value_in_unit(unit.radian)

    def restraint(self, x):
        """
        Restraints the angle between three atoms using:
        k_rho * (rho - rho_equ) ** 2
        with 
        k_rho   ... force constant
        rho     ... curennt angle
        rho_equ ... equilibrium angle

        Parameters
        -------
        x : torch.Tensor
            coordinates
        Returns
        -------
        e : torch.Tensor
        """

        assert(type(x) == torch.Tensor)

        # calculating angle using rho = arcos(x . y / |x||y|)
        # calculate scaled vector for bond_ij
        distance_ij = torch.norm(x[0][self.atom_j_idx] - x[0][self.atom_i_idx])
        direction_ij = (x[0][self.atom_j_idx] - x[0][self.atom_i_idx])
        bond_ij = direction_ij / distance_ij

        # calculate scaled vector for bond_kj
        distance_kj = torch.norm(x[0][self.atom_j_idx] - x[0][self.atom_k_idx])
        direction_kj = (x[0][self.atom_j_idx] - x[0][self.atom_k_idx])
        bond_kj = direction_kj / distance_kj

        current_angle = torch.acos(torch.dot(bond_ij, bond_kj) / distance_ij * distance_kj)  # in radian

        # x in angstrom
        e = (self.k/2) * (self.water_angle - current_angle.double())**2
        logging.debug('Angle harmonic restraint restraint_bias introduced: {:0.4f}'.format(e.item()))
        return e.to(device=self.device)


class BondFlatBottomRestraint(BondRestraint):

    def __init__(self,
                 sigma: unit.Quantity,
                 atom_i_idx: int,
                 atom_j_idx: int,
                 atoms: str,
                 active_at: int = -1):
        super().__init__(sigma, atom_i_idx, atom_j_idx, atoms, active_at)

    def restraint(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        -------
        x : torch.Tensor
            coordinates
        Returns
        -------
        e : torch.Tensor
        """
        assert(type(x) == torch.Tensor)
        # x in angstrom
        distance = torch.norm(x[0][self.atom_i_idx] - x[0][self.atom_j_idx])
        if distance <= self.lower_bound:
            e = (self.k/2) * (self.lower_bound - distance.double())**2
        elif distance >= self.upper_bound:
            e = (self.k/2) * (distance.double() - self.upper_bound)**2
        else:
            e = torch.tensor(0.0, dtype=torch.double, device=self.device)
        logging.debug('Flat bottom restraint_bias introduced: {:0.4f}'.format(e.item()))
        return e.to(device=self.device)


class BondHarmonicRestraint(BondRestraint):

    def __init__(self,
                 sigma: unit.Quantity,
                 atom_i_idx: int,
                 atom_j_idx: int,
                 atoms: str,
                 active_at: int = -1):

        super().__init__(sigma, atom_i_idx, atom_j_idx, atoms, active_at)

    def restraint(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        -------
        x : torch.Tensor
            coordinates
        Returns
        -------
        e : torch.Tensor
        """
        assert(type(x) == torch.Tensor)
        # x in angstrom
        distance = torch.norm(x[0][self.atom_i_idx] - x[0][self.atom_j_idx])
        e = (self.k/2) * (distance.double() - self.mean_bond_length)**2
        logging.debug('Harmonic restraint_bias introduced: {:0.4f}'.format(e.item()))
        return e.to(device=self.device)


class CenterFlatBottomRestraint(PointAtomRestraint):
    def __init__(self,
                 sigma: unit.Quantity,
                 point: unit.Quantity,
                 radius: unit.Quantity,
                 atom_idx: int,
                 active_at: int = -1):
        """
        Flat well restraint that becomes active when atom moves outside of radius.
        Parameters
        ----------
        sigma : float, unit'd
        point : np.array, unit'd
        radius : float, unit'd
        atom_idx : list
            list of atoms idxs
        active_at : int
            Integer to indicccate at which state the restraint is fully active. Either 0 (for 
            lambda 0), or 1 (for lambda 1) or -1 (always active)
        """

        assert(type(sigma) == unit.Quantity)
        assert(type(point) == unit.Quantity)
        super().__init__(sigma, point.value_in_unit(unit.angstrom), active_at)

        self.atom_idx = atom_idx
        self.cutoff_radius = radius.value_in_unit(unit.angstrom)

    def restraint(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        -------
        x : torch.Tensor
            coordinates
        Returns
        -------
        e : torch.Tensor
        """
        # x in angstrom
        assert(type(x) == torch.Tensor)
        distance = torch.norm(x[0][self.atom_idx] - self.point)
        if distance >= self.cutoff_radius:
            e = (self.k/2) * (distance.double() - self.cutoff_radius)**2
        else:
            e = torch.tensor(0.0, dtype=torch.double, device=self.device)
        logging.debug('Flat center bottom restraint_bias introduced: {:0.4f}'.format(e.item()))
        return e.to(device=self.device)


class CenterOfMassRestraint(PointAtomRestraint):

    def __init__(self, sigma: unit.Quantity,
                 point: unit.Quantity,
                 atom_idx: list,
                 atoms: str,
                 active_at: int = -1):
        """
        Center of mass restraint.

        Parameters
        ----------
        sigma : in angstrom
        point : np.array, unit'd
        atom_idx : list
            list of atoms idxs
        atoms: str
            Str of atoms to retrieve element information
        """
        assert(type(sigma) == unit.Quantity)
        assert(type(point) == unit.Quantity)
        super().__init__(sigma, point.value_in_unit(unit.angstrom), active_at)
        self.atom_idx = atom_idx
        logger.info('Center Of Mass restraint added.')

        self.mass_list = []
        for i in atom_idx:
            self.mass_list.append(mass_dict_in_daltons[atoms[i]])
        masses = np.array(self.mass_list)
        scaled_masses = masses / masses.sum()
        self.masses = torch.tensor(scaled_masses, dtype=torch.double, device=self.device, requires_grad=True)

    def _calculate_center_of_mass(self, x):
        """
        Calculates the center of mass.
        One assumption that we are making here is that the ligand is at the beginning of the 
        atom str and coordinate file.
        """
        ligand_x = x[0][:len(self.mass_list)].double()  # select only the ligand coordinates
        return torch.matmul(ligand_x.T, self.masses)

    def restraint(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        -------
        x : torch.Tensor
            coordinates
        Returns
        -------
        e : torch.Tensor
        """
        # x in angstrom
        assert(type(x) == torch.Tensor)

        com = self._calculate_center_of_mass(x)
        com_distance_to_point = torch.norm(com - self.point)
        e = (self.k/2) * (com_distance_to_point.sum() ** 2)
        return e.to(device=self.device)
