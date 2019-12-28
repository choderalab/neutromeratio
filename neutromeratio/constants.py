import sys

import torch
from openmmtools.constants import kB
from simtk import unit

platform = 'cpu'
temperature = 300 * unit.kelvin
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
pressure = 101325.0 * unit.pascal
gas_constant = 0.0019872036 * (unit.kilocalorie_per_mole / unit.kelvin)
water_hoh_angle = 104.5 * unit.degree

exclude_set = ['molDWRow_1004',
 'molDWRow_1088',
 'molDWRow_1109',
 'molDWRow_1110',
 'molDWRow_1114',
 'molDWRow_1115',
 'molDWRow_1116',
 'molDWRow_1117',
 'molDWRow_1118',
 'molDWRow_1119',
 'molDWRow_112',
 'molDWRow_1120',
 'molDWRow_1123',
 'molDWRow_1124',
 'molDWRow_1125',
 'molDWRow_1149',
 'molDWRow_1150',
 'molDWRow_1152',
 'molDWRow_1184',
 'molDWRow_1185',
 'molDWRow_1186',
 'molDWRow_1187',
 'molDWRow_1189',
 'molDWRow_1221',
 'molDWRow_1222',
 'molDWRow_1223',
 'molDWRow_1224',
 'molDWRow_1226',
 'molDWRow_1227',
 'molDWRow_1236',
 'molDWRow_1237',
 'molDWRow_1238',
 'molDWRow_1262',
 'molDWRow_1263',
 'molDWRow_1264',
 'molDWRow_1265',
 'molDWRow_1266',
 'molDWRow_1267',
 'molDWRow_1275',
 'molDWRow_1279',
 'molDWRow_1280',
 'molDWRow_1282',
 'molDWRow_1283',
 'molDWRow_1323',
 'molDWRow_1429',
 'molDWRow_1486',
 'molDWRow_1533',
 'molDWRow_1553',
 'molDWRow_1555',
 'molDWRow_1557',
 'molDWRow_1558',
 'molDWRow_1565',
 'molDWRow_1571',
 'molDWRow_1593',
 'molDWRow_1665',
 'molDWRow_1668',
 'molDWRow_1671',
 'molDWRow_178',
 'molDWRow_182',
 'molDWRow_201',
 'molDWRow_204',
 'molDWRow_508',
 'molDWRow_553',
 'molDWRow_557',
 'molDWRow_568',
 'molDWRow_569',
 'molDWRow_570',
 'molDWRow_571',
 'molDWRow_576',
 'molDWRow_577',
 'molDWRow_580',
 'molDWRow_581',
 'molDWRow_582',
 'molDWRow_585',
 'molDWRow_586',
 'molDWRow_587',
 'molDWRow_588',
 'molDWRow_603',
 'molDWRow_604',
 'molDWRow_605',
 'molDWRow_606',
 'molDWRow_611',
 'molDWRow_612',
 'molDWRow_615',
 'molDWRow_616',
 'molDWRow_617',
 'molDWRow_618',
 'molDWRow_621',
 'molDWRow_622',
 'molDWRow_623',
 'molDWRow_624',
 'molDWRow_636',
 'molDWRow_643',
 'molDWRow_648',
 'molDWRow_675',
 'molDWRow_73',
 'molDWRow_74',
 'molDWRow_758',
 'molDWRow_767',
 'molDWRow_768',
 'molDWRow_784',
 'molDWRow_82',
 'molDWRow_83',
 'molDWRow_836',
 'molDWRow_837',
 'molDWRow_839',
 'molDWRow_840',
 'molDWRow_952',
 'molDWRow_953',
 'molDWRow_955',
 'molDWRow_988',
 'molDWRow_989',
 'molDWRow_990',
 'molDWRow_991',
 'molDWRow_992',
 'molDWRow_993']
