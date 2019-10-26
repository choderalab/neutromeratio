import neutromeratio
from openmmtools.constants import kB
from simtk import unit
import numpy as np
import pickle
import mdtraj as md
import torchani
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import random, sys
from neutromeratio.constants import platform, device
import pandas as pd
from rdkit.Chem import AllChem
from neutromeratio.constants import temperature, gas_constant
from scipy.special import logsumexp

def get_conformer_rmsd(mols)->list:
    """
    Calculate conformer-conformer RMSD.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    rmsd = np.zeros((len(mols), len(mols)),
                    dtype=float)
    for i, ref_mol in enumerate(mols):
        for j, fit_mol in enumerate(mols):
            if i >= j:
                continue
            rmsd[i, j] = AllChem.GetBestRMS(ref_mol, fit_mol)
            rmsd[j, i] = rmsd[i, j]
    return rmsd

def calculate_weighted_energy(e_list):
    #G = -RT Σ ln exp(-G/RT)

    l = []
    for energy in e_list:
        v = ((-1) * (energy)) / (gas_constant * temperature)
        l.append(v)

    e_bw = (-1) * gas_constant * temperature * (logsumexp(l)) 
    return e_bw



# extract smiles
exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))
f = pickle.load(open('../data/results/qm_results.pickle', 'rb'))

# generate minimum dG in vacuum/solution
for env in ['vac', 'solv']:
    energies = dict()
    for name in exp_results:

        t1_smiles = exp_results[name]['t1-smiles']
        t2_smiles = exp_results[name]['t2-smiles']
        # t1
        mols= f[name][t1_smiles]['vac']
        tmp_e_t1 = []
        for m in mols:
            tmp_e_t1.append(neutromeratio.reduced_pot(float(m.GetProp('G')) * unit.kilocalorie_per_mole ))
        # t2
        mols= f[name][t2_smiles]['vac']
        tmp_e_t2 = []
        for m in mols:
            tmp_e_t2.append(neutromeratio.reduced_pot(float(m.GetProp('G')) * unit.kilocalorie_per_mole ))
        
        if len(tmp_e_t1) == 0 or len(tmp_e_t2) == 0:
            print(name)
            continue

        energies[name] = min(tmp_e_t2) - min(tmp_e_t1)

    df = pd.DataFrame().from_dict(energies, orient='index', columns=['energy [kT]']).transpose()
    df.to_pickle(f"../data/results/DFT_{env}_minimum_dG.pickle")


   
rmsd_threshold = 0.3

for env in ['vac', 'solv']:
    r_MM = dict()
    for name in exp_results:
        t1_smiles = exp_results[name]['t1-smiles']
        t2_smiles = exp_results[name]['t2-smiles']
        bw_energies = []
        for smiles in [t1_smiles, t2_smiles]:
            mols= f[name][smiles][env]
            tmp_e = []
            filtered_energies = []
            for m in mols:
                tmp_e.append((float(m.GetProp('G')) * unit.kilocalorie_per_mole ))
            
            if len(tmp_e) == 0:
                continue
                
            rmsd = get_conformer_rmsd(mols)
            sort = np.argsort(tmp_e)  # sort by increasing energy
            keep = []  # always keep lowest-energy conformer
            discard = []
            for i in sort:

                # always keep lowest-energy conformer
                if len(keep) == 0:
                    keep.append(i)
                    continue

                # get RMSD to selected conformers
                this_rmsd = rmsd[i][np.asarray(keep, dtype=int)]

                # discard conformers within the RMSD threshold
                if np.all(this_rmsd >= rmsd_threshold):
                    keep.append(i)
                else:
                    discard.append(i)

            for i in keep:
                filtered_energies.append(tmp_e[i])
                
            bw_energies.append(neutromeratio.reduced_pot(calculate_weighted_energy(filtered_energies)))
        if len(bw_energies) <= 1:
            continue
        e = bw_energies[1] - bw_energies[0]
        r_MM[name] = e
    
    df = pd.DataFrame().from_dict(r_MM, orient='index', columns=['energy [kT]']).transpose()
    df.to_pickle(f"../data/results/DFT_{env}_mining_minima_dG.pickle")
