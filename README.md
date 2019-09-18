neutromeratio
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/choderalab/neutromeratio.png)](https://travis-ci.org/choderalab/neutromeratio)
[![codecov](https://codecov.io/gh/choderalab/neutromeratio/branch/master/graph/badge.svg)](https://codecov.io/gh/choderalab/neutromeratio/branch/master)

Using neural net potentials to calculate tautomeric ratios.

### Introduction
Accurate calculations of tautomer ratios are hard. Quantum aspects of the proton transfer reaction and entropy changes introduced by the change in double bond positions present a particular theoretical challenge. Most of the published work on tautomer ratio calculations have phenomenological character and closer inspection reveal unjustified assumptions and/or cancelation of errors [1].  

A reliable method for the determination of tautomer ratios would ideally include conformational sampling in explicit water and the calculation of the potential energy with QM level accuracy. These two goals seem reasonable and within reach with the development of neural net potentials like ANI-1 that enable energy calculation with DFT level accuracy and force field like time costs [2]. This allows the exploration of the conformational degrees of freedom while ensuring high level QM energy calculations.

### Copyright

Copyright (c) 2019, Marcus Wieder & Josh Fass // MSKCC


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0.
