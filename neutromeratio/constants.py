import sys
import numpy as np

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
ani_energy_unit = hartree_to_kJ_mol * unit.kilojoule_per_mole  # simtk.unit doesn't have hartree?
nm_to_angstroms = (1.0 * distance_unit) / (1.0 * ani_distance_unit)

mass_dict_in_daltons = {'H': 1.0, 'C': 12.0, 'N': 14.0, 'O': 16.0}

bond_length_dict = {frozenset(['C', 'H']): 1.09 * unit.angstrom,
                    frozenset(['O', 'H']): 0.96 * unit.angstrom,
                    frozenset(['N', 'H']): 1.01 * unit.angstrom
                    }

conversion_factor_eV_to_kJ_mol = 96.485
conversion_factor_radian_to_degree = 57.2958
pressure = 101325.0 * unit.pascal
gas_constant = 0.0019872036 * (unit.kilocalorie_per_mole / unit.kelvin)
water_hoh_angle = 104.5 * unit.degree

mols_with_charge = ['molDWRow_400',
 'molDWRow_369',
 'molDWRow_651',
 'molDWRow_283',
 'molDWRow_626',
 'molDWRow_111',
 'molDWRow_286',
 'molDWRow_589',
 'molDWRow_284',
 'molDWRow_1004',
 'molDWRow_287',
 'molDWRow_282',
 'molDWRow_556',
 'molDWRow_955',
 'molDWRow_915',
 'molDWRow_412',
 'molDWRow_413',
 'molDWRow_914',
 'molDWRow_625',
 'molDWRow_1564',
 'molDWRow_285',
 'molDWRow_951',
 'molDWRow_994',
'molDWRow_287', # zwitterion
'molDWRow_400', # zwitterion
'molDWRow_412', # zwitterion
'molDWRow_412',  # zwitterion
'molDWRow_651',  # zwitterion
'molDWRow_369',  # zwitterion
]



exclude_set_ANI = [
               'molDWRow_1088', # sulfur
               'molDWRow_1109', #sulfur
               'molDWRow_1110', #sulfur, Cl
               'molDWRow_1114', # S, Cl
               'molDWRow_1115', # S, Cl
               'molDWRow_1116', #S
               'molDWRow_1117', #S
               'molDWRow_1118', #S
               'molDWRow_1119', #S 
               'molDWRow_112', #S
               'molDWRow_1120', #S
               'molDWRow_1123', #S
               'molDWRow_1124', #S
               'molDWRow_1125', #S
               'molDWRow_1149', #S
               'molDWRow_1150', #S
               'molDWRow_1152', #S
               'molDWRow_1184', #Cl
               'molDWRow_1185', #Cl
               'molDWRow_1186', #S
               'molDWRow_1187', #S
               'molDWRow_1189', #Cl
               'molDWRow_1221', #S
               'molDWRow_1222', #S
               'molDWRow_1223', #S
               'molDWRow_1224', #S
               'molDWRow_1226', #S
               'molDWRow_1227', #S
               'molDWRow_1236', #S
               'molDWRow_1237', #S
               'molDWRow_1238', #S
               'molDWRow_1260', #??????
               'molDWRow_1262', #Cl
               'molDWRow_1263', #Cl
               'molDWRow_1264', #F
               'molDWRow_1265', #S
               'molDWRow_1266', #F
               'molDWRow_1267', #Cl, S
               'molDWRow_1275', #Cl, S
               'molDWRow_1279', #Cl, S
               'molDWRow_1280', #Cl, S
               'molDWRow_1282', #Cl, S
               'molDWRow_1283', #Cl
               'molDWRow_1323', #S
               'molDWRow_1429', #S
               'molDWRow_1486', #S
               'molDWRow_1533', #S
               'molDWRow_1553', #S
               'molDWRow_1555', #F
               'molDWRow_1557', #S
               'molDWRow_1558', #S, F
               'molDWRow_1565', #S
               'molDWRow_1571', #S
               'molDWRow_1593', #S
               'molDWRow_1665', #S
               'molDWRow_1668', #S
               'molDWRow_1671', #S
               'molDWRow_178', #S
               'molDWRow_182', #S
               'molDWRow_201', #S
               'molDWRow_204', #S
               'molDWRow_508', #S
               'molDWRow_553', #Cl
               'molDWRow_557', #Cl
               'molDWRow_568', #S
               'molDWRow_569', #S
               'molDWRow_570', #F
               'molDWRow_571', #F
               'molDWRow_576', #I
               'molDWRow_577', #I
               'molDWRow_580', #Cl
               'molDWRow_581', #Br
               'molDWRow_582', #Br
               'molDWRow_585', #S
               'molDWRow_586', #S
               'molDWRow_587', #F
               'molDWRow_588', #F
               'molDWRow_603', #S
               'molDWRow_604', #S
               'molDWRow_605', #F
               'molDWRow_606', #F
               'molDWRow_611', #I
               'molDWRow_612', #I
               'molDWRow_615', #Cl
               'molDWRow_616', #Cl
               'molDWRow_617', #Br
               'molDWRow_618', #Br
               'molDWRow_621', #S
               'molDWRow_622', #S
               'molDWRow_623', #F
               'molDWRow_624', #F
               'molDWRow_636', #S
               'molDWRow_643', #Cl
               'molDWRow_648', #S
               'molDWRow_675', #S
               'molDWRow_73', #S
               'molDWRow_74', #S
               'molDWRow_758', #Cl
               'molDWRow_767', #S
               'molDWRow_768', #S
               'molDWRow_784', #S
               'molDWRow_82', #Cl
               'molDWRow_83', #Br
               'molDWRow_836', #S
               'molDWRow_837', #S
               'molDWRow_839', #S
               'molDWRow_840', #S
               'molDWRow_952', #Br
               'molDWRow_953', #Cl
               'molDWRow_955', #charge, Cl
               'molDWRow_988', #Br
               'molDWRow_989', #Br
               'molDWRow_990', #Cl
               'molDWRow_991', #Cl
               'molDWRow_992', #Cl
               'molDWRow_993', #I
                'molDWRow_1610', # Two hydrogens are moving -- not really a tautomer
                'molDWRow_1665', # Two hydrogens are moving -- not really a tautomer
                'molDWRow_1666', # Two hydrogens are moving -- not really a tautomer
                'molDWRow_1667', # Two hydrogens are moving -- not really a tautomer
                'molDWRow_1671', # Two hydrogens are moving -- not really a tautomer
                'molDWRow_1672', # Two hydrogens are moving -- not really a tautomer
                'molDWRow_1673', # Two hydrogens are moving -- not really a tautomer
                'molDWRow_1675', # Two hydrogens are moving -- not really a tautomer
                'molDWRow_1676', # Two hydrogens are moving -- not really a tautomer
               ]



exclude_set_B3LYP = [
               'molDWRow_576', #I
               'molDWRow_577', #I
               'molDWRow_611', #I
               'molDWRow_612', #I
               'molDWRow_955', #charge, Cl
               'molDWRow_993', #I
                'molDWRow_601'  # one conf in solution has a max-min of 16724 kcal/mol
                'molDWRow_37'  # one conf in solution has a max-min of 16724 kcal/mol
                'molDWRow_1610', # Two hydrogens are moving -- not really a tautomer
                'molDWRow_1665', # Two hydrogens are moving -- not really a tautomer
                'molDWRow_1666', # Two hydrogens are moving -- not really a tautomer
                'molDWRow_1667', # Two hydrogens are moving -- not really a tautomer
                'molDWRow_1671', # Two hydrogens are moving -- not really a tautomer
                'molDWRow_1672', # Two hydrogens are moving -- not really a tautomer
                'molDWRow_1673', # Two hydrogens are moving -- not really a tautomer
                'molDWRow_1675', # Two hydrogens are moving -- not really a tautomer
                'molDWRow_1676',  # Two hydrogens are moving -- not really a tautomer
                'molDWRow_1260',# molecules were drawn without Nitrogen in the database!
                'molDWRow_1261',# molecules were drawn without Nitrogen in the database!
                'molDWRow_1262',# molecules were drawn without Nitrogen in the database!
                'molDWRow_1263',# molecules were drawn without Nitrogen in the database!
                'molDWRow_1264',# molecules were drawn without Nitrogen in the database!
                'molDWRow_514',# wrong pyrimidine tautomer
                'molDWRow_515',# wrong pyrimidine tautomer
                'molDWRow_516',# wrong pyrimidine tautomer
                'molDWRow_517',# wrong pyrimidine tautomer
                'molDWRow_1587',# wrong structure 
               ]



# Imines are excluded from the tautomer_set_with_stereobonds since these don't need additional sampling
tautomer_set_with_stereobonds = ['molDWRow_1082',
 'molDWRow_1083',
 'molDWRow_113',
 'molDWRow_114',
 'molDWRow_115',
 'molDWRow_116',
 'molDWRow_117',
 'molDWRow_1182',
 'molDWRow_1183',
 'molDWRow_119',
 'molDWRow_120',
 'molDWRow_121',
 'molDWRow_122',
 'molDWRow_1232',
 'molDWRow_1234',
 'molDWRow_1235',
 'molDWRow_1240',
 'molDWRow_1243',
 'molDWRow_1254',
 'molDWRow_126',
 #'molDWRow_1260', # NOTE: stereobond is not kept rigid
 #'molDWRow_1261', # NOTE: stereobond is not kept rigid
 'molDWRow_1456',
 'molDWRow_1534',
 'molDWRow_1547',
 'molDWRow_1556',
 'molDWRow_1559',
 'molDWRow_1560',
 'molDWRow_1569',
 'molDWRow_251',
 'molDWRow_282',
 'molDWRow_283',
 'molDWRow_284',
 'molDWRow_285',
 'molDWRow_286',
 'molDWRow_298',
 'molDWRow_309',
 'molDWRow_37',
 'molDWRow_38',
 'molDWRow_401',
 'molDWRow_402',
 'molDWRow_415',
 'molDWRow_418',
 'molDWRow_46',
 'molDWRow_50',
 'molDWRow_507',
 'molDWRow_509',
 'molDWRow_51',
 'molDWRow_511',
 'molDWRow_512',
 'molDWRow_514',
 'molDWRow_516',
 'molDWRow_52',
 'molDWRow_521',
 'molDWRow_54',
 'molDWRow_649',
 'molDWRow_708',
 'molDWRow_735',
 'molDWRow_853',
 'molDWRow_973']

minimum_torsion_angles_in_degree = {
'molDWRow_1082' : {'idx' : [5,4,6,7], 'torsion' : np.linspace(180,0,21)},
 'molDWRow_1083': {'idx' : [5,4,6,7], 'torsion' : np.linspace(180,0,21)},
 'molDWRow_113':  {'idx' : [3,4,6,7], 'torsion' : np.linspace(70,-70,21)},
 'molDWRow_114': {'idx' : [9,8,1,2], 'torsion' : np.linspace(70,-70,21)},
 'molDWRow_115': {'idx' : [9,8,1,2], 'torsion' : np.linspace(-70,-180,21)},
 'molDWRow_116': {'idx' : [9,8,1,2], 'torsion' : np.linspace(-70,-180,21)},
 'molDWRow_117': {'idx' : [9,8,1,2], 'torsion' : np.linspace(-75,75,21)},
 'molDWRow_1182': {'idx' : [5,6,7,8], 'torsion' : np.linspace(180,0,21)},
 'molDWRow_1183': {'idx' : [4,3,2,1], 'torsion' : np.linspace(0,180,21)},
 'molDWRow_119': {'idx' : [9,8,1,2], 'torsion' : np.linspace(75,-75,21)},
 'molDWRow_120': {'idx' : [9,8,1,2], 'torsion' : np.linspace(75,-75,21)},
 'molDWRow_121': {'idx' : [9,8,1,2], 'torsion' : np.linspace(-75,75,21)},
 'molDWRow_122': {'idx' : [9,8,1,2], 'torsion' : np.linspace(-75,75,21)},
 'molDWRow_1232': {'idx' : [3,4,5,6], 'torsion' : np.linspace(180,0,21)},
 'molDWRow_1234': {'idx' : [3,4,5,6], 'torsion' : np.linspace(180,0,21)},
 'molDWRow_1235': {'idx' : [12,11,10,9], 'torsion' : np.linspace(180,0,21)},
 'molDWRow_1240': {'idx' : [0,1,2,12], 'torsion' : np.linspace(180,0,21)},
 'molDWRow_1243': {'idx' : [7,6,5,4], 'torsion' : np.linspace(180,0,21)}, # that one seems strange
 'molDWRow_1254': {'idx' : [7,6,5,4], 'torsion' : np.linspace(60,-60,21)},
 'molDWRow_126': {'idx' : [7,6,5,4], 'torsion' : np.linspace(75,-75,21)},
 #'molDWRow_1260': {?????}, # NOTE: stereobonds are not kept constant -- this is an imine that allows interconversion between cis and trans R=NR 
 #'molDWRow_1261': {????????},# https://chemistry.stackexchange.com/questions/26304/why-are-oxime-geometrical-isomers-stable
 'molDWRow_1456': {'idx' : [7,6,5,4], 'torsion' : np.linspace(75,-75,21)},
 'molDWRow_1534': [],
 'molDWRow_1547': [],
 'molDWRow_1556': [],
 'molDWRow_1559': [],
 'molDWRow_1560': [],
 'molDWRow_1569': [],
 'molDWRow_251': [],
 'molDWRow_282': [],
 'molDWRow_283': [],
 'molDWRow_284': [],
 'molDWRow_285': [],
 'molDWRow_286': [],
 'molDWRow_298': [],
 'molDWRow_309': [],
 'molDWRow_37': [],
 'molDWRow_38': [],
 'molDWRow_401': [],
 'molDWRow_402': [],
 'molDWRow_415': [],
 'molDWRow_418': [],
 'molDWRow_46': [],
 'molDWRow_50': [],
 'molDWRow_507': [],
 'molDWRow_509': [],
 'molDWRow_51': [],
 'molDWRow_511': [],
 'molDWRow_512': [],
 'molDWRow_514': [],
 'molDWRow_516': [],
 'molDWRow_52': [],
 'molDWRow_521': [],
 'molDWRow_54': [],
 'molDWRow_649': [],
 'molDWRow_708': [],
 'molDWRow_735': [],
 'molDWRow_853': [],
 'molDWRow_973': []
 }