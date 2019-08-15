import torch
from simtk import unit

from .constants import nm_to_angstroms


class Restraint():
    def __init__(self, heavy_atom_index, hydrogen_index):
        self.heavy_atom_index = heavy_atom_index
        self.hydrogen_index = hydrogen_index

    def forward(self, x):
        raise NotImplementedError


class FlatBottomRestraint(Restraint):
    def __init__(self, heavy_atom_index, hydrogen_index,
                 min_dist=0.8 * unit.angstrom,
                 max_dist=1.2 * unit.angstrom,
                 spring_constant=10):
        super().__init__(heavy_atom_index, hydrogen_index)
        self.min_dist_in_angstroms = min_dist.value_in_unit(unit.angstrom)
        self.max_dist_in_angstroms = max_dist.value_in_unit(unit.angstrom)
        self.spring_constant = spring_constant
        # TODO: units on spring_constant

    def forward(self, x):
        """Assumes x is in units of nanometers"""
        assert (len(x) == 1)  # TODO: assumes x is a [1, n_atoms, 3] tensor
        distance_in_angstroms = (
                torch.norm(x[0][self.hydrogen_index] - x[0][self.heavy_atom_index]) * nm_to_angstroms).double()

        left_penalty = (distance_in_angstroms < self.min_dist_in_angstroms) * (
                self.spring_constant * (self.min_dist_in_angstroms - distance_in_angstroms) ** 2)
        right_penalty = (distance_in_angstroms > self.max_dist_in_angstroms) * (
                self.spring_constant * (distance_in_angstroms - self.max_dist_in_angstroms) ** 2)
        return left_penalty + right_penalty


class HarmonicRestraint(Restraint):
    def __init__(self, heavy_atom_index, hydrogen_index,
                 eq_dist=1.0 * unit.angstrom,
                 spring_constant=10):
        super().__init__(heavy_atom_index, hydrogen_index)
        self.eq_dist_in_angstroms = eq_dist.value_in_unit(unit.angstrom)
        self.spring_constant = spring_constant
        # TODO: units on spring_constant

    def forward(self, x):
        assert (len(x) == 1)
        distance_in_angstroms = (
                torch.norm(x[0][self.hydrogen_index] - x[0][self.heavy_atom_index]) * nm_to_angstroms).double()
        penalty = (self.spring_constant * (distance_in_angstroms - self.eq_dist_in_angstroms) ** 2)
        return penalty
