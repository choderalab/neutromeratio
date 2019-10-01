from openmmtools.constants import kB
from simtk import unit
import sys
import torch

# this is a pointer to the module object instance itself.
# this = sys.modules[__name__]
# print(this)
# # we can explicitly make assignments on it 
# this.temperature = None
# this.platform = None
# this.kT = None
# this.device = None

# def initialize_variables(temperature, platform):
#     if (this.temperature is None):
#         # also in local function scope. no scope specifier like global is needed
#         if type(temperature) == unit.Quantity:
#             this.temperature = temperature
#             this.kT = kB * temperature
#         else:
#             this.temperature = temperature * unit.kelvin
#             this.kT = kB * temperature
#     else:
#         msg = "Temperature is already initialized to {0}."
#         raise RuntimeError(msg.format(this.temperature))

#     if (this.platform is None):
#         # also in local function scope. no scope specifier like global is needed
#         this.platform = platform
#         this.device = torch.device(platform)
#     else:
#         msg = "Platform is already initialized to {0}."
#         raise RuntimeError(msg.format(this.platform))

temperature = 300 * unit.kelvin
platform = 'cpu'
kT = kB * temperature
device = torch.device(platform)
torch.set_num_threads(2)

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

mass_dict_in_daltons = {'H': 1.0, 'C': 12.0, 'N': 14.0, 'O': 16.0}

bond_length_dict = {frozenset(['C', 'H']) : 1.09 * unit.angstrom,
                    frozenset(['O', 'H']) : 0.96 * unit.angstrom,
                    frozenset(['N', 'H']) : 1.01 * unit.angstrom
                    }

conversion_factor_eV_to_kJ_mol = 96.485