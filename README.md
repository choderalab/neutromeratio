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
```git clone https://github.com/choderalab/neutromeratio.git
cd neutromeratio
python setup.py install```

You will need to install a few dependencies --- all of them are defined in https://github.com/choderalab/neutromeratio/blob/master/devtools/conda-envs/test_env.yaml.


### Data

We provide data for the following preprint https://www.biorxiv.org/content/10.1101/2020.10.24.353318v1 .

## Dataset

The original data used for this study was taken from https://github.com/WahlOya/Tautobase. 
A subset of this was used to calculate relative free energies using a DFT approach, the molecules 
are shown here:
https://github.com/choderalab/neutromeratio/blob/dev-mw/data/b3lyp_tautobase_subset.txt
The subset of the subset above used for the calculations with the neural net potential is shown here:
https://github.com/choderalab/neutromeratio/blob/dev-mw/data/ani_tautobase_subset.txt

The text file includes the molecule name used in the manuscript, the SMILES for both tautomeric forms and the experimental ddG (in kcal/mol)

## Quantum chemistry data




### Copyright

Copyright (c) 2019, Marcus Wieder & Josh Fass // MSKCC


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0.
