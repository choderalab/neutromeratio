neutromeratio
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/neutromeratio.png)](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/neutromeratio)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/neutromeratio/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/neutromeratio/branch/master)

Using neural net potentials to calculate energies and nonequilibrium monte carlo to sample tautomeric states.

### Introduction
Accurate calculations of tautomer ratios are hard. Quantum aspects of the proton transfer reaction present a particular theoretical challenge. Most of the published work on tautomer ratio calculations have phenomenological character and closer inspection reveal unjustified assumptions and/or cancelation of errors [1].  

A reliable method for the determination of tautomer ratios would ideally include conformational sampling in explicit water and the calculation of the potential energy with QM level accuracy. These two goals seem reasonable and within reach with the development of neural net potentials like ANI-1 that enable energy calculation with DFT level accuracy and force field like time costs [2]. This allows the exploration of the conformational degrees of freedom while ensuring high level QM energy calculations.

This package contains code to run monte carlo (MC) and non-equilibrium candidate monte carlo simulations [3] (NCMC) using the ANI-1ccx potential to calculate work distributions for forward and reverse transformation between a set of tautomers and utilizes bennet’s acceptance ratio (BAR) to estimate the free energy difference between tautomer states. 

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

There are two types of restrains (flat bottom restraint and harmonic restraint) added to the heavy atom - hydrogen bond during the nonequilibrium protocoll in order to keep the hydrogen near its bonded partner while not fully coupled to the environment. These restraints are not contributing to the energy of the endpoints and only to the 'alchemical' part of the protocoll. At the endpoints only the bottom restraint is active. The two restraint are combined in the following way.

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{restraint}&space;=&space;(1-\lambda)&space;*&space;\text{bottom\_restraint}&space;&plus;&space;\lambda&space;*&space;\text{harmonic\_restraint}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{restraint}&space;=&space;(1-\lambda)&space;*&space;\text{bottom\_restraint}&space;&plus;&space;\lambda&space;*&space;\text{harmonic\_restraint}" title="\text{restraint} = (1-\lambda) * \text{bottom\_restraint} + \lambda * \text{harmonic\_restraint}" /></a>

### Preliminary results

#### Running Langevin dynamics using ANI-1ccx

We generated equilibrium samples (using openMM and gaff2) for the two tautomeric forms of mol298 and used openMM, ANI-1ccx and psi4 (wB97X/6-31g*) to generate energy histograms and plot the energy as a function of the timestep.

![hist_energy_comparision_t1](https://user-images.githubusercontent.com/31651017/63472076-54871c00-c471-11e9-81dc-e348c8faef8d.png)![hist_energy_comparision_t2](https://user-images.githubusercontent.com/31651017/63472089-610b7480-c471-11e9-8a45-afc3efbd2b1f.png)![plot_energy_comparision_t1](https://user-images.githubusercontent.com/31651017/63472110-68cb1900-c471-11e9-9995-27357fafc0d8.png)![plot_energy_comparision_t2](https://user-images.githubusercontent.com/31651017/63472121-708abd80-c471-11e9-92d8-9a4307581bd3.png)



#### MC

Results for the MC/NCMC protocol are shown for a single tautomer transformation:
![molDWRow_590_tautomers](https://user-images.githubusercontent.com/31651017/63469748-3e765d00-c46b-11e9-8c3d-63185eac93d8.png)


Results of the MC protocol that switches tautomer states in a single step from an equilibrium sample.
![mc_work_distribution](https://user-images.githubusercontent.com/31651017/63470778-c8bfc080-c46d-11e9-9135-1d1b7c972796.png)

Bar estimate of the work: 2.53 +- 0.73 kcal/mol.

#### NCMC

The distance between the hydrogen acceptor (in green) and the hydrogen donor (in blue) and the tautomer-hydrogen as well as the applied restraints (in read) are shown as a function of the protocol length.

![molDWRow_298_from_t1_to_t2_NCMC_distance_run_nr_0](https://user-images.githubusercontent.com/31651017/63471363-36b8b780-c46f-11e9-8a6c-68613e56a73e.png)

The forward/reverse work distribution of NCMC protocol is:

![work_hist_molDWRow_590_NCMC](https://user-images.githubusercontent.com/31651017/63469829-7382af80-c46b-11e9-9bef-b57a4dc6b51c.png)

The BAR estimate of the work is -0.33 +- 0.11 kcal/mol. Seperate QM calculations of the same molecule using orca, B3LYP/aug-cc-pVTZ resulted in a reference energy difference of -3.36 kcal/mol.

The cummulative standard deviation for the NCMC protocol is shown below.
![cwork_stddev_molDWRow_590_NCMC](https://user-images.githubusercontent.com/31651017/63469900-91e8ab00-c46b-11e9-8a8c-7601e1a7c69b.png)

The work values and standard deviation for each protocol step for the transformation of tautomer 1 to tautomer 2:
![work_summary_molDWRow_590_from_t1_to_t2_NCMC](https://user-images.githubusercontent.com/31651017/63470118-0facb680-c46c-11e9-93be-b902fae4a9ad.png) ![work_stddev_molDWRow_590_from_t1_to_t2_NCMC](https://user-images.githubusercontent.com/31651017/63470160-2521e080-c46c-11e9-9235-2080c2c57313.png)

The work values and standard deviation for each protocol step for the transformation of tautomer 2 to tautomer 1:
![work_summary_molDWRow_590_from_t2_to_t1_NCMC](https://user-images.githubusercontent.com/31651017/63470285-89dd3b00-c46c-11e9-8d5a-0e075c2e7a53.png)![work_stddev_molDWRow_590_from_t2_to_t1_NCMC](https://user-images.githubusercontent.com/31651017/63470301-92357600-c46c-11e9-818f-240470e9cf8c.png)



### Some movies:
A trajectory without restraints around the hydrogen - heavy atom bond:
https://drive.google.com/file/d/1_yFzTneNdiWb2LD5AMgjOYOboKaPgZ9N/view?usp=sharing

A trajectory from the current protocol, with restraints:
https://drive.google.com/file/d/1BieyQ7odaljaOQGVHGV7LAP7VrLMeqoQ/view?usp=sharing


## What do you need to start

The jupyter notebook notebooks/NCMC-tautomers.ipynb can be used to start the NCMC protocol (1000 steps for the perturbation, 150 repetitions of the protocol) for a set of tautomer SMILES. So all you need for starting the protocl are a set of SMILES strings that can be interconverted by moving a single hydrogen. As long as the two SMILES describe two tautomeric states of the same small molecule the hydrogen that needs to move, the heavy atom donor that looses the hydrogen and the heavy atom acceptor that receives the hydorgen are automatically detected and the protocol needs no further input.


## Implementation details

The calculation of the energies in the current torchANI implementation (https://github.com/aiqm/torchani) is in float32. This lead initially to precission issiues with increased NCMC protocl length. We are using float64 for summing up the energies for the neural net potentials avoiding this issue.

```python
class DoubleAniModel(torchani.nn.ANIModel):

    def forward(self, species_aev):
        # change dtype
        species, aev = species_aev
        species_ = species.flatten()
        present_species = torchani.utils.present_species(species)
        aev = aev.flatten(0, 1)

        output = torch.full_like(species_, self.padding_fill,
                                dtype=torch.float32)
        for i in present_species:
            mask = (species_ == i)
            input_ = aev.index_select(0, mask.nonzero().squeeze())
            output.masked_scatter_(mask, self[i](input_).squeeze())
        
        output = output.view_as(species)
        return species, self.reducer(output.double(), dim=1)
```

Equilibrium samples are generated using Langevin Dynamics in a pure python implementation. A starting position is provided and equilibrium samples are generated and returned.

```python
class LangevinDynamics(object):

    def __init__(self, atom_list:str, temperature:int, force:ANI1_force_and_energy):
        self.force = force
        self.temperature = temperature
        self.atom_list = atom_list


    def run_dynamics(self, x0:np.ndarray, 
                    n_steps:int = 100,
                    stepsize:unit.quantity.Quantity = 1 * unit.femtosecond,
                    collision_rate:unit.quantity.Quantity = 10/unit.picoseconds,
                     progress_bar:bool = False,
            ):
        """Unadjusted Langevin dynamics.

        Parameters
        ----------
        x0 : array of floats, unit'd (distance unit)
            initial configuration
        force : callable, accepts a unit'd array and returns a unit'd array
            assumes input is in units of distance
            output is in units of energy / distance
        n_steps : integer
            number of Langevin steps
        stepsize : float > 0, in units of time
            finite timestep parameter
        collision_rate : float > 0, in units of 1/time
            controls the rate of interaction with the heat bath
        progress_bar : bool
            use tqdm to show progress bar

        Returns
        -------
        traj : [n_steps + 1 x dim] array of floats, unit'd
            trajectory of samples generated by Langevin dynamics

        """

        assert(type(x0) == unit.Quantity)
        assert(type(stepsize) == unit.Quantity)
        assert(type(collision_rate) == unit.Quantity)
        assert(type(self.temperature) == unit.Quantity)

        # generate mass arrays
        mass_dict_in_daltons = {'H': 1.0, 'C': 12.0, 'N': 14.0, 'O': 16.0}
        masses = np.array([mass_dict_in_daltons[a] for a in self.atom_list]) * unit.daltons
        sigma_v = np.array([unit.sqrt(kB * self.temperature / m) / speed_unit for m in masses]) * speed_unit
        v0 = np.random.randn(len(sigma_v),3) * sigma_v[:,None]
        # convert initial state numpy arrays with correct attached units
        x = np.array(x0.value_in_unit(distance_unit)) * distance_unit
        v = np.array(v0.value_in_unit(speed_unit)) * speed_unit

        # traj is accumulated as a list of arrays with attached units
        traj = [x]
        # dimensionless scalars
        a = np.exp(- collision_rate * stepsize)
        b = np.sqrt(1 - np.exp(-2 * collision_rate * stepsize))

        # compute force on initial configuration
        F = self.force.calculate_force(x)
        trange = range(n_steps)
        if progress_bar:
            trange = tqdm(trange)
        for _ in trange:

            # v
            v += (stepsize * 0.5) * F / masses[:,None]
            # r
            x += (stepsize * 0.5) * v
            # o
            v = (a * v) + (b * sigma_v[:,None] * np.random.randn(*x.shape))
            # r
            x += (stepsize * 0.5) * v
            F = self.force.calculate_force(x)
            # v
            v += (stepsize * 0.5) * F / masses[:,None]

            norm_F = np.linalg.norm(F)
            # report gradient norm
            if progress_bar:
                trange.set_postfix({'|force|': norm_F})
            # check positions and forces are finite
            if (not np.isfinite(x).all()) or (not np.isfinite(norm_F)):
                print("Numerical instability encountered!")
                return traj
            traj.append(x)
        return traj
```

The real workhorse of the NCMC protocoll is the AlchemicalANI class. It enables the linear scaling of 
the energies between two atomic environment vectors as a function of a coupling parameter.

```python

class LinearAlchemicalANI(AlchemicalANI):
    def __init__(self, alchemical_atom):
        """Scale the indirect contributions of alchemical atoms to the energy sum by
        linearly interpolating, for other atom i, between the energy E_i^0 it would compute
        in the _complete absence_ of the alchemical atoms, and the energy E_i^1 it would compute
        in the _presence_ of the alchemical atoms.
        (Also scale direct contributions, as in DirectAlchemicalANI)
        """

        super().__init__(alchemical_atom)      
        self.neural_networks = self._load_model_ensemble(self.species, self.ensemble_prefix, self.ensemble_size)

    def forward(self, species_coordinates):
        # for now only allow one alchemical atom

        # LAMBDA = 1: fully interacting
        # species, AEVs of fully interacting system
        species, coordinates, lam = species_coordinates
        species, aevs = self.aev_computer(species_coordinates[:-1])
        # neural net output given these AEVs
        nn_1 = self.neural_networks((species, aevs))[1]
        E_1 = self.energy_shifter((species, nn_1))[1]
        
        if float(lam) == 1.0:
            E = E_1
        else:
            # LAMBDA = 0: fully removed
            # species, AEVs of all other atoms, in absence of alchemical atoms
            mod_species = torch.cat((species[:, :self.alchemical_atom],  species[:, self.alchemical_atom+1:]), dim=1)
            mod_coordinates = torch.cat((coordinates[:, :self.alchemical_atom],  coordinates[:, self.alchemical_atom+1:]), dim=1) 
            mod_aevs = self.aev_computer((mod_species, mod_coordinates))[1]
            # neural net output given these modified AEVs
            nn_0 = self.neural_networks((mod_species, mod_aevs))[1]
            E_0 = self.energy_shifter((species, nn_0))[1]
            E = (lam * E_1) + ((1 - lam) * E_0)
        return species, E
```




### Copyright

Copyright (c) 2019, Marcus Wieder & Josh Fass // MSKCC


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0.
