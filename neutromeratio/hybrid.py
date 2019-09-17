import copy
from .mcmc import MC_Mover
from .ani import ANI1_force_and_energy 
import logging
import mdtraj as md
import torch
from simtk import unit
from ase import Atom, Atoms
import torchani
from .constants import device
from .restraints import Restraint

logger = logging.getLogger(__name__)

def generate_hybrid_structure(ani_input:dict, tautomer_transformation:dict):
    """
    Generates a hybrid structure between two tautomers. The heavy atom frame is kept but a
    hydrogen is added to the tautomer acceptor heavy atom. 
    Keys are added to the ani_input dict and tautomer_transformation dict in place - 
    NOTHING IS RETURNED!:
    All entries regarding hybrid topology are added with 
    keyword 'hybrid' at the beginning. Restraints are 
    automatically added.
    ani_input['hybrid_atoms'] = ani_input['ligand_atoms'] + 'H'
    ani_input['hybrid_coords'] = hybrid_coord
    ani_input['hybrid_topolog'] = hybrid_top
    ani_input['hybrid_restraints'] = [Restraint1, Restraint2]

    tautomer_transformation['donor_hydrogen_idx'] = tautomer_transformation['hydrogen_idx']
    tautomer_transformation['acceptor_hydrogen_idx'] = len(ani_input['hybrid_atoms']) -1
    Parameters
    ----------
    ani_input : dict
    tautomer_transformation : dict
    """

    # add hybrid atoms
    ani_input['hybrid_atoms'] = ani_input['ligand_atoms'] + 'H'
    atoms = ani_input['hybrid_atoms']

    # generate 3D coordinates for hybrid atom
    model = torchani.models.ANI1ccx()
    model = model.to(device)

    energy_function = ANI1_force_and_energy(
                                          model = model,
                                          atoms = atoms
                                         )
    
    energy_function.use_pure_ani1ccx = True
    # generate MC mover to get new hydrogen position
    hydrogen_mover = MC_Mover(tautomer_transformation['donor_idx'], 
                            tautomer_transformation['hydrogen_idx'], 
                            tautomer_transformation['acceptor_idx'],
                            ani_input['ligand_atoms'])


    min_e = 100 * unit.kilocalorie_per_mole
    min_coordinates = None

    # from the multiple conformations in ani_input['ligand_coords'] we are taking a single
    # coordinate set (the first one) and add the hydrogen 
    for _ in range(500):
        hybrid_coord = hydrogen_mover._move_hydrogen_to_acceptor_idx(ani_input['ligand_coords'][0], override=False)
        e = energy_function.calculate_energy(hybrid_coord, lambda_value=1.0)
        if e < min_e:
            min_e = e
            min_coordinates = hybrid_coord 
    
    ani_input['min_e'] = min_e

    # add indixes for the hydrogens that are active at labda 0 (donor_hydrogen_idx)
    # and at lambda 1 (acceptor_hydrogen_idx)
    tautomer_transformation['donor_hydrogen_idx'] = tautomer_transformation['hydrogen_idx']
    tautomer_transformation['acceptor_hydrogen_idx'] = len(ani_input['hybrid_atoms']) -1
    ani_input['hybrid_coords'] = min_coordinates

    # generate hybrid restraints
    # donor restraints are active at lambda = 1 because then donor hydrogen is turend off and vica versa
    r1 = Restraint(0.1 * unit.angstrom, tautomer_transformation['donor_hydrogen_idx'], tautomer_transformation['donor_idx'], atoms=atoms, active_at_lambda=1)
    r2 = Restraint(0.1 * unit.angstrom, tautomer_transformation['acceptor_hydrogen_idx'], tautomer_transformation['acceptor_idx'], atoms=atoms, active_at_lambda=0)
    ani_input['hybrid_restraints'] = [r1, r2]

    # add to mdtraj ligand topology a new hydrogen
    hybrid_top = copy.deepcopy(ani_input['ligand_topology'])
    dummy_atom = hybrid_top.add_atom('H', md.element.hydrogen, hybrid_top.residue(-1))
    hybrid_top.add_bond(hybrid_top.atom(tautomer_transformation['acceptor_idx']), dummy_atom)
    # save new top in ani_input
    ani_input['hybrid_topology'] = hybrid_top

    # generate an ASE topology for the hybrid mol to minimze later 
    atom_list = []
    for e, c in zip(ani_input['hybrid_atoms'], ani_input['hybrid_coords']):
        c_list = (c[0].value_in_unit(unit.angstrom), c[1].value_in_unit(unit.angstrom), c[2].value_in_unit(unit.angstrom)) 
        atom_list.append(Atom(e, c_list))
    mol = Atoms(atom_list)
    ani_input['ase_hybrid_mol'] = mol





