import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from simtk import unit
import subprocess
from .constants import hartree_to_kJ_mol

# https://orcaforum.kofo.mpg.de/viewtopic.php?f=8&t=3860&p=15652&hilit=solvation#p15652
# http://www.ccl.net/chemistry/resources/messages/2011/12/01.001-dir/
# https://orcaforum.kofo.mpg.de/viewtopic.php?f=8&t=2431&p=23498&hilit=calculate+solvation+energy#p23498


def generate_orca_script_for_solvation_free_energy(
    mol: Chem.Mol, conformer_id: int = 0
) -> str:
    """Returns a psi4 Molecule object instance for a single conformer from a rdkit mol object.

    Parameters
    ----------
    mol : Chem.Mol
        rdkit mol object
    conformer_id : int
        specifies the conformer to use

    Returns
    -------
    mol : psi4.core.Molecule
        a psi4 molecule object instance
    """

    assert type(mol) == Chem.Mol or type(mol) == Chem.PropertyMol.PropertyMol
    atoms = mol.GetAtoms()

    cpcm_str = """
%cpcm
smd true
solvent "WATER"
end\n\n"""

    header_str = "! B3LYP 6-31G*  \n"
    xyz_string_str = f"*xyz 0 1\n"
    for _, atom in enumerate(atoms):
        pos = mol.GetConformer(conformer_id).GetAtomPosition(atom.GetIdx())
        xyz_string_str += f"{atom.GetSymbol()} {pos.x:.7f} {pos.y:.7f} {pos.z:.7f}\n"
    xyz_string_str += "*"

    orca_input = header_str + cpcm_str + xyz_string_str

    return orca_input


def generate_orca_script_for_gas_phase(mol: Chem.Mol, conformer_id: int = 0) -> str:
    """Returns a psi4 Molecule object instance for a single conformer from a rdkit mol object.

    Parameters
    ----------
    mol : Chem.Mol
        rdkit mol object
    conformer_id : int
        specifies the conformer to use

    Returns
    -------
    mol : psi4.core.Molecule
        a psi4 molecule object instance
    """

    assert type(mol) == Chem.Mol or type(mol) == Chem.PropertyMol.PropertyMol
    atoms = mol.GetAtoms()

    header_str = "! B3LYP 6-31G*  \n"
    xyz_string_str = f"*xyz 0 1\n"
    for _, atom in enumerate(atoms):
        pos = mol.GetConformer(conformer_id).GetAtomPosition(atom.GetIdx())
        xyz_string_str += f"{atom.GetSymbol()} {pos.x:.7f} {pos.y:.7f} {pos.z:.7f}\n"
    xyz_string_str += "*"

    orca_input = header_str + xyz_string_str

    return orca_input


def run_orca(file_path):

    from subprocess import Popen, PIPE

    p = Popen(
        [
            "/home/mwieder/Work/Software/orca_4_0_1_2_linux_x86-64_openmpi202/orca",
            file_path,
        ],
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
    )
    output, err = p.communicate(b"input data that is passed to subprocess' stdin")
    rc = p.returncode

    if rc != 0:
        print(err)
        raise RuntimeError(f"Orca returned: {rc}. Aborting.")

    return rc, output, err
