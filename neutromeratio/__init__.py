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
from .tautomers import get_tautomer_transformation, from_mol_to_ani_input
from .utils import reduced_pot, add_solvent, write_pdb, generate_rdkit_mol
from .vis import display_mol, generate_nglview_object
from .mcmc import Instantaneous_MC_Mover, NonequilibriumMC
from .equilibrium import LangevinDynamics, use_precalculated_md_and_performe_mc
from .ani import ANI1_force_and_energy, LinearAlchemicalANI, LinearAlchemicalSingleTopologyANI
from .restraints import Restraint
from .hybrid import generate_hybrid_structure
# TODO: generating waterbox

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
