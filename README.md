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

Data and scripts to reproduce figures are provided here:
[![DOI](https://zenodo.org/badge/341970689.svg)](https://zenodo.org/badge/latestdoi/341970689)

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
