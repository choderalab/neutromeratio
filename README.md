neutromeratio
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/neutromeratio.png)](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/neutromeratio)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/neutromeratio/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/neutromeratio/branch/master)

Using neural net potentials to calculate energies and nonequilibrium monte carlo to sample tautomeric states.

### Introduction
Accurate calculations of tautomer ratios are hard. Quantum aspects of the proton transfer reaction present a particular theoretical challenge. Most of the published work on tautomer ratio calculations have phenomenological character and closer inspection reveal unjustified assumptions and/or cancelation of errors [1].  

A reliable method for the determination of tautomer ratios would ideally include conformational sampling in explicit water and the calculation of the potential energy with QM level accuracy. These two goals seem reasonable and within reach with the development of neural net potentials like ANI-1 that enable energy calculation with DFT level accuracy and force field like time costs [2]. This enables the exploration of the conformational degrees of freedom while ensuring high level QM energy calculations.

This package contains code to run non-equilibrium candidate monte carlo simulations [3] (NCMC) using the ANI-1 potetnial to calculate work distributions for forward and reverse transformation between a set of tautomers and utilizes bennet’s acceptance ratio (BAR) to estimate the free energy difference between the work distributions. 

### Theoretical background

NCMC constructs a non equilibrium protocol that achieves high acceptance rates with short correlation times. Instead of instantaneous proposing a new tautomer state from an equilibrium sample it uses a coupling parameter to construct a non equilibrium process with incremental changes to slowly transform tautomer 1 to tautomer 2. After each incremental perturbation the system is allowed to relax -- in such a way highly efficient proposals can be constructed.

The current implementation of the protocoll does not actually accept any of these proposals. The work is recorded and used to calculate the free energy difference between the two states.

The propagation along the coupling parameter for the nonequilibrium protocol has three stages that contribute to the final work value: 

1. Decoupling of the hydrogen
2. Moving the hydrogen to a new position
3. Coupling the hydrogen


An instantaneous MC protocol would perform all three described steps in a single propagation step while the nonequilibrium protocol uses small incremental perturbations for (1) and (3).

The (de)coupling is performed by linearly scaling the energy contribution of the hydrogen that changes its binding partner to the total energy of the system. This is possible because the total energy of a molecule (Et) is computed as a sum over the output of neural net potential for each individual atom:

<a href="https://www.codecogs.com/eqnedit.php?latex=E_{t}&space;=&space;\sum^{\text{all&space;atoms}}_{i}&space;E_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?E_{t}&space;=&space;\sum^{\text{all&space;atoms}}_{i}&space;E_{i}" title="E_{t} = \sum^{\text{all atoms}}_{i} E_{i}" /></a>


There are still contributions of a specific atom i to all other atoms if removed from the total sum of neural net potentials, as eqn. (2), (3) and eqn (4) in [2] contain sums over neighboring atoms. 
There are two ways to scale these ‘indirect’ contributions of a given atom i to its neighbors: either linearly scaling the atomic environment vectors between an environment with atom i and an environment without atom i. The downside of such an approach is that a linear scaling of the atomic environment vector does not necessarily result in a linear scaling of the energy. Another possibility is to scale the energy directly between an atomic environment vector with atom i and without atom i which would guarantee that the energy change of the system is linear in respect to a coupling parameter.  

We decided on linearly scaling the total energy. That means at any given lambda there are two <a href="https://www.codecogs.com/eqnedit.php?latex=E_{t}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?E_{t}" title="E_{t}" /></a> values: <a href="https://www.codecogs.com/eqnedit.php?latex=E_{t,&space;\lambda=0}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?E_{t,&space;\lambda=0}" title="E_{t, \lambda=0}" /></a> is calculated with the unmodified atomic environment vector and contains the sum over all atoms. <a href="https://www.codecogs.com/eqnedit.php?latex=E_{t,&space;\lambda=1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?E_{t,&space;\lambda=1}" title="E_{t, \lambda=1}" /></a> is calculated with the modified atomic environment vector for which the atom that will be modified is removed. To scale the energies depending on lambda the following equation is used for the final energy:

<a href="https://www.codecogs.com/eqnedit.php?latex=E_{t}&space;=&space;(\lambda&space;*&space;E_{t,&space;\lambda=0})&space;&plus;&space;((1&space;-&space;\lambda)&space;*&space;E_{t,&space;\lambda=1})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?E_{t}&space;=&space;(\lambda&space;*&space;E_{t,&space;\lambda=0})&space;&plus;&space;((1&space;-&space;\lambda)&space;*&space;E_{t,&space;\lambda=1})" title="E_{t} = (\lambda * E_{t, \lambda=0}) + ((1 - \lambda) * E_{t, \lambda=1})" /></a>

The work values for step (1) and step (3) of the protocol (decoupling and coupling) is the sum over the dE along the protocol. For step (2) the work calculations will be discussed in more detail.

The acceptance ratio for proposing to move from configuration x (initial coordinates) in thermodynamic state A (tautomeric state 1) to configuration x’ (proposed coordinates) in thermodynamic state B (tautomeric state 2) is calculated as follows.

<a href="https://www.codecogs.com/eqnedit.php?latex=R_{A\rightarrow&space;B}(x\rightarrow&space;{x}')&space;=&space;\frac{p_{B}{x}'}{p_{A}{x}}&space;\frac{g_{B\rightarrow&space;A}{({x}'\rightarrow&space;x})}{g_{A\rightarrow&space;B}{(x\rightarrow&space;{x}'})}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?R_{A\rightarrow&space;B}(x\rightarrow&space;{x}')&space;=&space;\frac{p_{B}{x}'}{p_{A}{x}}&space;\frac{g_{B\rightarrow&space;A}{({x}'\rightarrow&space;x})}{g_{A\rightarrow&space;B}{(x\rightarrow&space;{x}'})}" title="R_{A\rightarrow B}(x\rightarrow {x}') = \frac{p_{B}{x}'}{p_{A}{x}} \frac{g_{B\rightarrow A}{({x}'\rightarrow x})}{g_{A\rightarrow B}{(x\rightarrow {x}'})}" /></a>

The first ratio describes the probability of <a href="https://www.codecogs.com/eqnedit.php?latex={x}'" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{x}'" title="{x}'" /></a> in thermodynamic state B divided by the probability of <a href="https://www.codecogs.com/eqnedit.php?latex={x}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{x}" title="{x}" /></a> in thermodynamic state A. This ratio is simply the difference in energy of the system in the initial and proposed position. The second ratio describes the probability density functions for the proposal process that generates configurations in tautomer state B given a configuration in tautomer state A and vice versa. The proposal density function <a href="https://www.codecogs.com/eqnedit.php?latex=g(B\rightarrow&space;A)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?g(B\rightarrow&space;A)" title="g(B\rightarrow A)" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=g(A\rightarrow&space;B)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?g(A\rightarrow&space;B)" title="g(A\rightarrow B)" /></a> depends on the equilibrium bond length between the heavy atom and hydrogen. Since the hydrogen can move between acceptor and donor atoms with different elements and the equilibrium bond length can change the proposal density is not symmetric and the calculation can not be omitted.

We introduce a flat bottom potential around the heavy atom donor and hydrogen bond to avoid ‘losing’ the hydrogen during the (de)coupling. The restraint contributes to the work only in the alchemical part of the protocol when the hydrogen losses interactions with the bonded heavy atom but not at the endpoints. Eqn(3) is evaluated with the energy contribution of the restraint.




### Copyright

Copyright (c) 2019, Marcus Wieder & Josh Fass // MSKCC


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0.
