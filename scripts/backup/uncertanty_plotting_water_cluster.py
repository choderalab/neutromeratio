import neutromeratio
from openmmtools.constants import kB
from simtk import unit
import numpy as np
import pickle
import mdtraj as md
import matplotlib.pyplot as plt
import sys
import torch
from neutromeratio.parameter_gradients import FreeEnergyCalculator
from neutromeratio.constants import kT, device, hartree_to_kJ_mol
import torchani
import nglview
from tqdm import tqdm
from glob import iglob

f = '/data/chodera/wiederm'
file_list = [f for f in iglob('**/*', recursive=True) if os.path.isfile(f)]


mdtraj = md.load_dcd(dcd, top=pdb)
atoms = [a.element.symbol for a in mdtraj.topology.atoms]
model = torchani.models.ANI1ccx()

energy_function = neutromeratio.ANI1_force_and_energy(model = model,
                                          atoms = atoms,
                                          mol = None,
                                          use_pure_ani1ccx=True)




n_models = energy_function.model.ensemble_size
xyz_in_angstroms = mdtraj.xyz * 10
coordinates = torch.tensor(xyz_in_angstroms)
n_snapshots = len(coordinates)
species = energy_function.species


atoms = [a.element.symbol for a in mdtraj.topology.atoms]
model = torchani.models.ANI1ccx()

energy_function = neutromeratio.ANI1_force_and_energy(model = model,
                                          atoms = atoms,
                                          mol = None,
                                          use_pure_ani1ccx=True)




n_models = energy_function.model.ensemble_size

xyz_in_angstroms = mdtraj[::100].xyz * 10
coordinates = torch.tensor(xyz_in_angstroms)
n_snapshots = len(coordinates)

species = energy_function.species
species_list = species.tolist()[0]
n_atoms = len(species_list)
mod_species = torch.stack([species[0]] * len(coordinates))

print('n_snapshots: {}, n_atoms: {}, n_models: {}'.format(n_snapshots, n_atoms, n_models))

print('species.shape', mod_species.shape)
print('coordinates.shape', coordinates.shape)
_, aevs = model.aev_computer((mod_species, coordinates))

raw_energies = np.zeros((n_snapshots, n_atoms, n_models))
for i in tqdm(range(n_atoms)):
    for j in range(n_models):
        raw_energies[:, i, j] = model.neural_networks[j][species_list[i]].forward(
            aevs[:, i, :]).detach().flatten() * hartree_to_kJ_mol


fig = plt.figure(figsize=[7,7], dpi=300)
fontsize=15

pos = plt.imshow(-np.mean(raw_energies, axis=2).transpose(), cmap='Blues')
plt.title('Ensemble discrepancy by atom in kcal/mol')
plt.ylabel('MD snapshost', size=fontsize)
plt.ylabel('atom idx', size=fontsize)
plt.colorbar(pos)
plt.tight_layout()
plt.savefig()

bad_inds = []
m = np.mean(raw_energies, axis=2).transpose()
for idx in range(n_atoms):
    if any(e  < -100 for e in list(m[idx])):
        bad_inds.append(idx)


"""raw_energies is an array of shape (n_snapshots, n_atoms, n_models), assumed in hartree
bad_inds is an integer array of atom indices that we want to color -- all other atoms will be grey
"""
plt.plot(
    (raw_energies.mean(-1) - raw_energies.mean(-1)[0]),
c='grey', alpha=0.1)


mean = (raw_energies.mean(-1)[:,bad_inds] - raw_energies.mean(-1)[0,bad_inds]) 
stddev = (raw_energies).std(-1)[:,bad_inds]
for i in range(len(bad_inds)):
    plt.plot(np.arange(len(raw_energies)), mean[:,i])
    plt.fill_between(np.arange(len(raw_energies)),
                    mean[:,i] - stddev[:,i], mean[:,i] + stddev[:,i], alpha=0.1)


plt.ylabel('atomic contribution to total energy (kJ/mol)\n(centered at contribution at $t=0$)')
plt.xlabel('MD snapshot')
start_of_sentence = 'each oxygen that switches into spurious geometry'
end_of_sentence = 'contributes very favorably to the total energy'
end_of_sentence = 'is highly "controversial"'
plt.title('{}\n{}'.format(start_of_sentence, end_of_sentence))

plt.tight_layout()
