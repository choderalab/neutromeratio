"""
neutromeratio
Using neural net potentials to sample tautomer states.
"""

import logging

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)1s()] %(message)s"
logging.basicConfig(format=FORMAT, datefmt="%d-%m-%Y:%H:%M", level=logging.INFO)

# Add imports here
from .tautomers import Tautomer
from .utils import reduced_pot, write_pdb, generate_rdkit_mol
from .vis import display_mol, generate_nglview_object
from .mcmc import Instantaneous_MC_Mover, NonequilibriumMC
from .equilibrium import LangevinDynamics, use_precalculated_md_and_performe_mc
from .ani import ANI_force_and_energy, AlchemicalANI2x, AlchemicalANI1ccx
from .analysis import (
    _remove_hydrogens,
    prune_conformers,
    calculate_weighted_energy,
    compare_confomer_generator_and_trajectory_minimum_structures,
)

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
