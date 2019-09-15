"""
neutromeratio
Using neural net potentials to sample tautomer states.
"""

import logging

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)1s()] %(message)s"
logging.basicConfig(format=FORMAT,
    datefmt='%d-%m-%Y:%H:%M',
    level=logging.INFO)

# Add imports here
from .conformations import generate_conformations_from_mol
from .tautomers import get_tautomer_transformation
from .utils import display_mol, from_mol_to_ani_input, reduced_pot, add_solvent, write_pdb, generate_nglview_object, generate_rdkit_mol
from .mcmc import Instantaneous_MC_Mover, NonequilibriumMC
from .equilibrium import LangevinDynamics, use_precalculated_md_and_performe_mc
from .ani import ANI1_force_and_energy
from .hybrid import generate_hybrid_structure
from .restraints import Restraint
# TODO: generating waterbox

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
