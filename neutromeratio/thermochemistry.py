import ase
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
from simtk import unit
from constants import temperature

def get_thermo_correction(ase_mol:ase.Atoms) -> simtk.unit.quantity.Quantity :
    """
    Returns the thermochemistry correction.
    Parameters
    ----------
    mol : ase.Atoms
    Returns
    -------
    gibbs_energy_correction : 
    """

    assert(type(ase_mol) == ase.Atoms)
    vib = Vibrations(ase_mol)
    vib.run()
    vib_energies = vib.get_energies()
    thermo = IdealGasThermo(vib_energies=vib_energies,
                            atoms=ase_mol,
                            geometry='nonlinear',
                            symmetrynumber=1, spin=0)
    
    G = thermo.get_gibbs_energy(temperature=temperature.value_in_unit(unit.kelvin), pressure=101325.)

    return (G * 96.485) * unit.kilojoule_per_mole # eV * conversion_factor(eV to kJ/mol)

