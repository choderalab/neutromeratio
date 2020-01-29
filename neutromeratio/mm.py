# TODO: script --> function called for each tautomer pair
# TODO: save relativefep results
# TODO: save endpoint samples
# TODO: vacuum --> droplet
# TODO: instantaneous --> noneq endpoint correction

import numpy as np
from tqdm import tqdm

# openff imports
import openforcefield
from openforcefield.topology import Molecule
from openforcefield.typing.engines.smirnoff import ForceField
forcefield = ForceField('openff-1.0.0.offxml')
print(openforcefield._version.get_versions())

# openmm imports
from simtk import openmm as mm
from simtk.openmm import LangevinIntegrator
from simtk.openmm.app import Simulation

# simulation settings
from simtk import unit
from constants import temperature
from openmmtools.constants import kB

kBT = kB * temperature
stepsize = 1 * unit.femtosecond
collision_rate = 1 / unit.picosecond

n_lambdas = 20

# relative free energy setup tools
from perses.rjmc.topology_proposal import TopologyProposal
from perses.annihilation.relative import HybridTopologyFactory
from perses.annihilation.lambda_protocol import LambdaProtocol

# analysis imports
from pymbar import MBAR, EXP

# ANI imports
import torchani
import torch

# define a tautomer pair t1,t2 and parameterize both
t1_smiles = "Oc1nccnc1"
t2_smiles = "O=C1NC=CN=C1"
t1 = Molecule.from_smiles(t1_smiles, hydrogens_are_explicit=False)
t2 = Molecule.from_smiles(t2_smiles, hydrogens_are_explicit=False)

# create vacuum simulation
def create_sim(molecule):
    platform = mm.Platform.getPlatformByName('Reference')
    integrator = LangevinIntegrator(temperature, collision_rate, stepsize)
    
    topology = molecule.to_topology()
    system = forcefield.create_openmm_system(topology)

    sim = Simulation(topology, system, integrator, platform=platform)
    
    molecule.generate_conformers()
    sim.context.setPositions(molecule.conformers[0])
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(temperature)
    return sim

def get_positions(sim):
    return sim.context.getState(getPositions=True).getPositions(asNumpy=True)

def get_energy(sim, positions):
    sim.context.setPositions(positions)
    return sim.context.getState(getEnergy=True).getPotentialEnergy()
    
t1_sim = create_sim(t1)
t2_sim = create_sim(t2)

def collect_samples(sim, n_samples=100, n_steps_per_sample=1000):
    samples = []
    for _ in tqdm(range(n_samples)):
        sim.step(n_steps_per_sample)
        samples.append(get_positions(sim))
    return samples

# collect end-state samples
t1_samples = collect_samples(t1_sim)
t2_samples = collect_samples(t2_sim)

# relative free energy setup
old_system, old_topology = t1_sim.system, t1_sim.topology.to_openmm()
new_system, new_topology = t2_sim.system, t2_sim.topology.to_openmm()

n_atoms = new_topology.getNumAtoms()

new_to_old_atom_map = {i:i for i in range(n_atoms) if t1.atoms[i].atomic_number > 1} # map heavy atoms to themselves

topology_proposal = TopologyProposal(
    new_topology=new_topology, new_system=new_system,
    old_topology=old_topology, old_system=old_system,
    old_chemical_state_key='t1',
    new_chemical_state_key='t2',
    new_to_old_atom_map=new_to_old_atom_map,
)

hybrid_factory = HybridTopologyFactory(
    topology_proposal=topology_proposal,
    current_positions=t1.conformers[0],
    new_positions=t2.conformers[0],
    use_dispersion_correction=False, 
    softcore_LJ_v2=True, # TODO: double-check if recommended
    interpolate_old_and_new_14s=True, # TODO: double-check if recommended
)

platform = mm.Platform.getPlatformByName('CPU')
integrator = LangevinIntegrator(temperature, collision_rate, stepsize)
    
hybrid_sim = Simulation(
    hybrid_factory.hybrid_topology,
    hybrid_factory.hybrid_system,
    integrator,
    platform,
)

protocol = LambdaProtocol()

def set_lambda(hybrid_sim, master_lambda=0.0):
    hybrid_sim_params = set(dict(hybrid_sim.context.getParameters()).keys())
    for key in protocol.functions:
        if key in hybrid_sim_params:
            hybrid_sim.context.setParameter(key, protocol.functions[key](master_lambda))

def reset_conditions(hybrid_sim):
    hybrid_sim.context.setPositions(hybrid_factory.hybrid_positions)
    hybrid_sim.context.setVelocitiesToTemperature(temperature)

lambdas = np.linspace(0,1,n_lambdas)

all_samples = []
for lam in lambdas:
    reset_conditions(hybrid_sim)
    set_lambda(hybrid_sim, lam)
    
    hybrid_sim.minimizeEnergy()
    
    hybrid_sim.step(1000) # TODO: expose this constant as a parameter
    
    all_samples.append(collect_samples(hybrid_sim, n_steps_per_sample=100)) # TODO: expose this constant as a parameter

# compute free energy difference using MBAR
joined_samples = list(all_samples[0])
for l in all_samples[1:]:
    joined_samples.extend(l)

u_kn = np.zeros((len(lambdas), len(joined_samples)))
for i, lam in enumerate(lambdas):
    set_lambda(hybrid_sim, lam)
    
    for j, x in enumerate(joined_samples):
        u_kn[i][j] = get_energy(hybrid_sim, x) / kBT

mbar = MBAR(u_kn, N_k=list(map(len, all_samples)))

# DeltaG(t1_MM --> t2_MM)
MM_predicted_deltaG = mbar.f_k[-1] * kBT
print('DeltaG(t1_MM --> t2_MM): {:.3f} kcal/mol'.format(MM_predicted_deltaG /unit.kilocalorie_per_mole))

# end-point corrections
t1_energies = [get_energy(t1_sim, s) for s in t1_samples]
t2_energies = [get_energy(t2_sim, s) for s in t2_samples]

model = torchani.models.ANI1ccx()
species_string = ''.join([a.element.symbol for a in t1.atoms])
species = model.species_to_tensor(species_string).unsqueeze(0)

def compute_ani_energy(samples):
    coordinates = torch.tensor([sample / unit.angstrom for sample in samples], dtype=torch.float32)
    energy = model((torch.stack([species[0]] * len(samples)), coordinates))
    return energy.energies.detach().numpy() * 627.5 * unit.kilocalorie_per_mole # convert from hartree to kcal/mol

t1_ani_energies = compute_ani_energy(t1_samples)
t2_ani_energies = compute_ani_energy(t2_samples)

t1_mm_energy_array = np.array([e / unit.kilocalorie_per_mole for e in t1_energies]) * unit.kilocalorie_per_mole
t2_mm_energy_array = np.array([e / unit.kilocalorie_per_mole for e in t2_energies]) * unit.kilocalorie_per_mole


t1_endpoint_reduced_works = (t1_ani_energies - t1_mm_energy_array) / kBT
t2_endpoint_reduced_works = (t2_ani_energies - t2_mm_energy_array) / kBT

# DeltaG(t1_MM --> t1_ANI)
t1_endpoint_correction = EXP(t1_endpoint_reduced_works)[0] * kBT
# DeltaG(t2_MM --> t2_ANI)
t2_endpoint_correction = EXP(t2_endpoint_reduced_works)[0] * kBT

endpoint_correction = t2_endpoint_correction - t1_endpoint_correction
print('MM --> ANI endpoint correction: {:.3f} kcal/mol'.format(endpoint_correction / unit.kilocalorie_per_mole))

endpoint_corrected_deltaG = MM_predicted_deltaG + endpoint_correction

print('endpoint-corrected DeltaG(t1_ANI --> t2_ANI): {:.3f} kcal/mol'.format(endpoint_corrected_deltaG / unit.kilocalorie_per_mole))
