neutromeratio
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/choderalab/neutromeratio.png)](https://travis-ci.org/choderalab/neutromeratio)
[![codecov](https://codecov.io/gh/choderalab/neutromeratio/branch/master/graph/badge.svg)](https://codecov.io/gh/choderalab/neutromeratio/branch/master)

Using neural net potentials to calculate tautomeric ratios.

### Introduction
Accurate calculations of tautomer ratios are hard. Quantum aspects of the proton transfer reaction and entropy changes introduced by the change in double bond positions present a particular theoretical challenge. Most of the published work on tautomer ratio calculations have phenomenological character and closer inspection reveal unjustified assumptions and/or cancelation of errors.  

A reliable method for the determination of tautomer ratios would ideally include conformational sampling in explicit water and the calculation of the potential energy with QM level accuracy. These two goals seem reasonable and within reach with the development of neural net potentials like ANI-1 that enable energy calculation with DFT level accuracy and force field like time costs. This allows the exploration of the conformational degrees of freedom while ensuring high level QM energy calculations.


### Installation

install `neutromeratio` master via `git clone` and `python setup install`:
```
git clone https://github.com/choderalab/neutromeratio.git
cd neutromeratio
python setup.py install
```

You will need to install a few dependencies --- all of them are defined in https://github.com/choderalab/neutromeratio/blob/master/devtools/conda-envs/test_env.yaml.


# Data

We provide data for the following preprint https://www.biorxiv.org/content/10.1101/2020.10.24.353318v3 .

## Dataset

The original data used for this study was taken from https://github.com/WahlOya/Tautobase. 
A subset of this was used to calculate relative free energies using a DFT approach, the molecules 
are shown here:
https://github.com/choderalab/neutromeratio/blob/master/data/b3lyp_tautobase_subset.txt
The subset of the subset above used for the calculations with the neural net potential is shown here:
https://github.com/choderalab/neutromeratio/blob/master/data/ani_tautobase_subset.txt

The text file includes the molecule name used in the manuscript, the SMILES for both tautomeric forms and the experimental ddG (in kcal/mol).

## Quantum chemistry data

The full QM data is deposited in https://github.com/choderalab/neutromeratio/blob/master/data/results/QM/qm_results_final.pickle .

The pickle file contains a single dictionary (of dictionaries), with the molecule names as keys.
```
r = pickle.load(open('qm_results_final.pickle', 'rb'))
r['SAMPLmol2']
```

returns:

```
{'OC1=CC=C2C=CC=CC2=N1': {'solv': [mol1, mol2, ...],
  'vac': [mol1, mol2, ...]},
 'O=C1NC2=C(C=CC=C2)C=C1': {'solv': [mol1, mol2, ...],
  'vac': [mol1, mol2, ...]}}
``` 
.

For a single system (e.g. `'SAMPLmol2'`) the two tautomer molecules are identified with the SMILES string (e.g. `'OC1=CC=C2C=CC=CC2=N1'`), and the envrionment (e.g. `'solv'`).
Using these three keys (e.g. `r['SAMPLmol2]['OC1=CC=C2C=CC=CC2=N1']['solv])` one gets a list or rdkit molecules, each in the optimized 3D conformation.
The molecule contains properties that can be acces via `.GetProp()`.
The relevant properties are:

`'G'` ... the gibbs free energy in the specified environment calculated with RRHO and B3LYP/aug-cc-pVTZ

`'E_B3LYP_pVTZ'` ... electronic energy evaluated on this conformation using B3LYP/aug-cc-pVTZ

`'E_B3LYP_631G_gas'` ... electronic energy evaluated on this conformation using B3LYP/6-31G(d)

`'E_B3LYP_631G_solv'` ... electronic energy evaluated on this conformation using B3LYP/6-31G(d)/SMD
`'H'` ... the enthalpy in the specified environment calculated with RRHO and B3LYP/aug-cc-pVTZ 
`'graph_automorphism'` ... the number of graph automorphism. This is used for the calculation of `RT ln(D)` where `D` is this number.


## ANI data

### RRHO ANI dataset

The raw data for this data set is saved in https://github.com/choderalab/neutromeratio/blob/master/data/results/ANI1ccx_RRHO.pickle.
This pickle file contains a dictionary that can be queried using the molecule names as key.
For each molecule there are additional keys: `'t1-energies'`, `'t2-energies'`, `'t1-confs'` and `'t2-confs'`.
t1 and t2 correspond to the naming of the tautomers in the original dataset.

``t1-energies`` contains a list with the gibbs free energies after energy minimization,  `'t1-confs'` contains a rdkit molecule with the conformations after minimization.


### Alchemical free energy dataset

Results for 5 independent runs are stored here: https://github.com/choderalab/neutromeratio/tree/master/data/results/AFE_ANI1ccx_vacuum.
Each of the 5 csv files contains a per line three values: the name of the tautomer system, ddG [kcal/mol] and dddG [kcal/mol].

### Optimization

The retraining results are located here (including the log file and best parameter set):
https://github.com/choderalab/neutromeratio/tree/master/data/retraining

The used script is here:
https://github.com/choderalab/neutromeratio/blob/master/data/retraining/parameter_opt.py

# How to generate the data 

There are four notebooks in https://github.com/choderalab/neutromeratio/tree/master/notebooks.
These generate all the data shown in the manuscript and in the Supplementary Information.

# How to use neutromeration


## Running alchemical free energy calculations for tautomers

It is pretty easy --- the relevant bash script is here:
https://github.com/choderalab/neutromeratio/blob/master/scripts/generate_samples_and_analyse_results.sh .
In the bash script, relevant parameters are defined at the very beginning:
```
SMILES1='OC1=CC=C2C=CC=CC2=N1'
SMILES2='O=C1NC2=C(C=CC=C2)C=C1'
name='test_mol' # defines where the output directory name
base_path="./" # where do you want to save the ouput files -> the ouput will have the form ${base_path}/${name}
potential_name='ANI1ccx' # which potential do you want to use? (ANI1ccx, ANI1x, ANI2x)
n_steps=10000 # nr of steps (dt = 0.5fs)
env='vacuum' 
```
 
To simulate a new tautomer pair you have to update the SMILES strings. The results are saved in {base_path}/{name} and the simulations will be performed for {n_steps}. A reasonable value for n_steps should be above 100,000.
This script automatically uses ANI1ccx and returns the results with the native ANI1ccx parameters and the best performing parameter set from the parameter fitting described here: https://www.biorxiv.org/content/10.1101/2020.10.24.353318v3




### Copyright

Copyright (c) 2019, Marcus Wieder & Josh Fass // MSKCC


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0.
