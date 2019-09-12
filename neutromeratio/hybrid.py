import copy
from .mcmc import MC_Mover
from .ani import ANI1_force_and_energy, LinearAlchemicalANI 
import logging
import mdtraj as md
import torch
from simtk import unit
from ase import Atom, Atoms

logger = logging.getLogger(__name__)

def generate_hybrid_structure(ani_input:dict, tautomer_transformation:dict, ANI1_force_and_energy:ANI1_force_and_energy):
    """
    Generates a hybrid structure between two tautomers. The heavy atom frame is kept but a
    hydrogen is added to the tautomer acceptor heavy atom. 
    Keys are added to the ani_input dict and tautomer_transformation dict:
    ani_input['hybrid_atoms'] = ani_input['ligand_atoms'] + 'H'
    ani_input['hybrid_coords'] = hybrid_coord
    ani_input['hybrid_topolog'] = hybrid_top
    tautomer_transformation['donor_hydrogen_idx'] = tautomer_transformation['hydrogen_idx']
    tautomer_transformation['acceptor_hydrogen_idx'] = len(ani_input['hybrid_atoms']) -1
    Parameters
    ----------
    ani_input : dict
    tautomer_transformation : traj
    ANI1_force_and_energy : ANI1_force_and_energy
    """
    platform = 'cpu'
    device = torch.device(platform)
    model = LinearAlchemicalANI(alchemical_atoms=[], ani_input={}, device=device, pbc=False)
    model = model.to(device)
    torch.set_num_threads(2)

    ani_input['hybrid_atoms'] = ani_input['ligand_atoms'] + 'H'

    energy_function = ANI1_force_and_energy(device = device,
                                          model = model,
                                          atom_list = ani_input['hybrid_atoms'],
                                          platform = platform,
                                          tautomer_transformation = None)
    # TODO: check type consistency: here tautomer_transformation=None, but default is {}

    # generate MC mover to get new hydrogen position
    hydrogen_mover = MC_Mover(tautomer_transformation['donor_idx'], 
                            tautomer_transformation['hydrogen_idx'], 
                            tautomer_transformation['acceptor_idx'],
                            ani_input['ligand_atoms'])


    min_e = 100 * unit.kilocalorie_per_mole
    min_coordinates = None

    # from the multiple conformations in ani_input['ligand_coords'] we are taking a single
    # coordinate set (the first one) and add the hydrogen 
    for _ in range(10):
        hybrid_coord = hydrogen_mover._move_hydrogen_to_acceptor_idx(ani_input['ligand_coords'][0], override=False)
        e = energy_function.calculate_energy(hybrid_coord)
        if e < min_e:
            min_e = e
            min_coordinates = hybrid_coord 
    
    ani_input['min_e'] = min_e
    tautomer_transformation['donor_hydrogen_idx'] = tautomer_transformation['hydrogen_idx']
    tautomer_transformation['acceptor_hydrogen_idx'] = len(ani_input['hybrid_atoms']) -1
    ani_input['hybrid_coords'] = min_coordinates

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





