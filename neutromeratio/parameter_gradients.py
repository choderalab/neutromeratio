# TODO: gradient of MBAR_estimated free energy difference w.r.t. model parameters

import logging

import numpy as np
import torch
from pymbar import MBAR
from pymbar.timeseries import detectEquilibration
from simtk import unit
from tqdm import tqdm
from glob import glob
from neutromeratio.ani import ANI1_force_and_energy
from neutromeratio.constants import hartree_to_kJ_mol, device, platform, kT, exclude_set_ANI, mols_with_charge
import torchani, torch
import os
import neutromeratio
import pickle
import mdtraj as md

logger = logging.getLogger(__name__)

class FreeEnergyCalculator():
    def __init__(self,
                 ani_model: ANI1_force_and_energy,
                 ani_trajs: list,
                 potential_energy_trajs: list,
                 lambdas: list,
                 n_atoms: int,
                 per_atom_thresh: unit.Quantity = 0.5*unit.kilojoule_per_mole,
                 max_snapshots_per_window=50,
                 ):
        """
        Uses mbar to calculate the free energy difference between trajectories.
        Parameters
        ----------
        ani_model : AlchemicalANI
            model used for energy calculation
        ani_trajs : list
            trajectories 
        potential_energy_trajs : list
            energy trace of trajectories
        lambdas : list
            all lambda states
        n_atoms : int
            number of atoms
        per_atom_thresh : unit'd
            exclude snapshots where ensemble stddev in energy / n_atoms exceeds this threshold
            if per_atom_tresh == -1 ignores ensemble stddev

        """
        K = len(lambdas)
        assert (len(ani_trajs) == K)
        assert (len(potential_energy_trajs) == K)
        logging.info(f"Per atom threshold used for filtering: {per_atom_thresh}")
        self.ani_model = ani_model
        self.potential_energy_trajs = potential_energy_trajs  # for detecting equilibrium
        self.lambdas = lambdas
        self.ani_trajs = ani_trajs
        self.n_atoms = n_atoms

        N_k, snapshots, used_lambdas = self.remove_confs_with_high_stddev(max_snapshots_per_window, per_atom_thresh)

        # end-point energies
        lambda0_e = [self.ani_model.calculate_energy([x/unit.angstrom] * unit.angstrom, lambda_value=0.0).energy_tensor for x in tqdm(snapshots)]
        lambda1_e = [self.ani_model.calculate_energy([x/unit.angstrom] * unit.angstrom, lambda_value=1.0).energy_tensor for x in tqdm(snapshots)]

        def get_mix(lambda0, lambda1, lam=0.0):
            return (1 - lam) * np.array(lambda0) + lam * np.array(lambda1)

        logger.info('Nr of atoms: {}'.format(n_atoms))

        u_kn = np.stack(
            [get_mix(lambda0_e, lambda1_e, lam) for lam in sorted(used_lambdas)]
        )
        self.mbar = MBAR(u_kn, N_k)
        self.snapshots = snapshots

    def remove_confs_with_high_stddev(self, max_snapshots_per_window: int, per_atom_thresh: unit.Quantity):
        """
        Removes conformations with ensemble energy stddev per atom above a given threshold.
        Parameters
        ----------
        max_snapshots_per_window : int
            maximum number of conformations per lambda window
        per_atom_thresh : float
            per atom stddev threshold - by default this is set to 0.5 kJ/mol/atom
        """

        def calculate_stddev(snapshots):
            # calculate energy, restraint_bias and stddev for endstates, extract stddev
            logger.info('Calculating stddev')
            lambda0_stddev = [self.ani_model.calculate_energy([x/unit.angstrom] * unit.angstrom, lambda_value=0.0).stddev/kT for x in tqdm(snapshots)]
            lambda1_stddev = [self.ani_model.calculate_energy([x/unit.angstrom] * unit.angstrom, lambda_value=1.0).stddev/kT for x in tqdm(snapshots)]

            return np.array(lambda0_stddev), np.array(lambda1_stddev)

        def compute_linear_ensemble_bias(current_stddev, per_atom_thresh):
            # calculate the total energy stddev threshold based on the provided per_atom_thresh
            # and the number of atoms
            total_thresh = ((per_atom_thresh / kT) * self.n_atoms)
            logger.info(f"Per system treshold: {total_thresh}")
            # if stddev for a given conformation < total_thresh => 0.0
            # if stddev for a given conformation > total_thresh => stddev - total_threshold
            linear_ensemble_bias = np.maximum(0, current_stddev - (total_thresh))
            return linear_ensemble_bias

        def compute_last_valid_ind(linear_ensemble_bias):
            # return the idx of the first entry that is above 0.0
            if np.sum(linear_ensemble_bias) < 0.01:  # means all can be used
                logger.info('Last valid ind: -1')
                return -1
            elif np.argmax(np.cumsum(linear_ensemble_bias) > 0) == 0:  # means nothing can be used
                logger.info('Last valid ind: 0')
                return 0
            else:
                idx = np.argmax(np.cumsum(linear_ensemble_bias) > 0) - 1
                logger.info(f"Last valid ind: {idx}")
                return idx  # means up to idx can be used

        ani_trajs = {}
        further_thinning = 10
        for lam, traj, potential_energy in zip(self.lambdas, self.ani_trajs, self.potential_energy_trajs):
            # detect equilibrium
            #equil, g = detectEquilibration(potential_energy)[:2]
            # thinn snapshots and return max_snapshots_per_window confs
            #snapshots = list(traj[int(len(traj)/2):].xyz * unit.nanometer)[::further_thinning]
            snapshots = list(traj.xyz * unit.nanometer)[:max_snapshots_per_window]
            ani_trajs[lam] = snapshots
            logger.info(f"Snapshots per lambda: {len(snapshots)}")

        last_valid_inds = {}
        logger.info(f"Looking through {len(ani_trajs)} lambda windows")
        for lam in sorted(ani_trajs.keys()):
            if per_atom_thresh < 0. * unit.kilojoule_per_mole:
                last_valid_ind = -1
            else:
                logger.info(f"Calculating stddev and penalty for lambda: {lam}")
                # calculate endstate stddev for given confs
                lambda0_stddev, lambda1_stddev = calculate_stddev(ani_trajs[lam])
                # scale for current lam
                current_stddev = (1 - lam) * lambda0_stddev + lam * lambda1_stddev
                linear_ensemble_bias = compute_linear_ensemble_bias(current_stddev, per_atom_thresh)
                last_valid_ind = compute_last_valid_ind(linear_ensemble_bias)
            last_valid_inds[lam] = last_valid_ind

        snapshots = []
        self.snapshot_mask = []
        N_k = []
        lambdas_with_usable_samples = []
        # lambbdas with usable samples applied to usable conformations
        for lam in sorted(list(last_valid_inds.keys())):
            logger.info(f"lam: {lam}")
            if last_valid_inds[lam] > 5:
                lambdas_with_usable_samples.append(lam)
                new_snapshots = ani_trajs[lam][0:last_valid_inds[lam]]
                logger.info(
                    f"For lambda {lam}: {len(new_snapshots)} snaphosts below treshold out of {len(ani_trajs[lam])}")
                snapshots.extend(new_snapshots)
                N_k.append(len(new_snapshots))
                self.snapshot_mask.append(len(new_snapshots))
            elif last_valid_inds[lam] == -1:  # -1 means all can be used
                lambdas_with_usable_samples.append(lam)
                new_snapshots = ani_trajs[lam][:]
                logger.info(
                    f"For lambda {lam}: {len(new_snapshots)} snaphosts below treshold out of {len(ani_trajs[lam])}")
                snapshots.extend(new_snapshots)
                N_k.append(len(new_snapshots))
                self.snapshot_mask.append(len(new_snapshots))

            else:
                self.snapshot_mask.append(0)
                logger.info(f"For lambda {lam}: ZERO snaphosts below treshold out of {len(ani_trajs[lam])}")
        logger.info(f"Nr of snapshots considered for postprocessing: {len(snapshots)}")
        return N_k, snapshots, lambdas_with_usable_samples

    @property
    def free_energy_differences(self):
        """matrix of free energy differences"""
        return self.mbar.getFreeEnergyDifferences(return_dict=True)['Delta_f']

    @property
    def free_energy_difference_uncertainties(self):
        """matrix of asymptotic uncertainty-estimates accompanying free energy differences"""
        return self.mbar.getFreeEnergyDifferences(return_dict=True)['dDelta_f']

    @property
    def end_state_free_energy_difference(self):
        """DeltaF[lambda=1 --> lambda=0]"""
        results = self.mbar.getFreeEnergyDifferences(return_dict=True)
        return results['Delta_f'][0, -1], results['dDelta_f'][0, -1]

    def compute_perturbed_free_energies(self, u_ln, u0_stddev, u1_stddev):
        """compute perturbed free energies at new thermodynamic states l"""
        assert (type(u_ln) == torch.Tensor)

        def torchify(x):
            return torch.tensor(x, dtype=torch.double, requires_grad=True, device=device)

        states_with_samples = torch.tensor(self.mbar.N_k > 0, device=device)
        N_k = torch.tensor(self.mbar.N_k, dtype=torch.double, requires_grad=True, device=device)
        f_k = torchify(self.mbar.f_k)
        u_kn = torchify(self.mbar.u_kn)

        log_q_k = f_k[states_with_samples] - u_kn[states_with_samples].T
        # TODO: double check that torch.logsumexp(x + torch.log(b)) is the same as scipy.special.logsumexp(x, b=b)
        A = log_q_k + torch.log(N_k[states_with_samples])
        log_denominator_n = torch.logsumexp(A, dim=1)

        B = - u_ln - log_denominator_n
        return - torch.logsumexp(B, dim=1)

    def form_u_ln(self):

        # bring list of unit'd coordinates in [N][K][3] * unit shape
        coordinates = [sample / unit.angstrom for sample in self.snapshots] * unit.angstrom
        
        # TODO: vectorize!
        decomposed_energy_list_lamb0 = self.ani_model.calculate_energy(coordinates, lambda_value=0)      
        u_0 = decomposed_energy_list_lamb0.energy_tensor
        u0_stddev = decomposed_energy_list_lamb0.stddev

        # TODO: vectorize!
        decomposed_energy_list_lamb1 = self.ani_model.calculate_energy(coordinates, lambda_value=1)     
        u_1 = decomposed_energy_list_lamb1.energy_tensor
        u1_stddev = decomposed_energy_list_lamb1.stddev

        u_ln = torch.stack([u_0, u_1])
        return u_ln,  u0_stddev, u1_stddev

    def compute_free_energy_difference(self):
        u_ln, u0_stddev, u1_stddev = self.form_u_ln()
        f_k = self.compute_perturbed_free_energies(u_ln, u0_stddev, u1_stddev)
        return f_k[1] - f_k[0]



def tweak_parameters(name:str = 'SAMPLmol2', data_path:str = "../data/"):

    def parse_lambda_from_dcd_filename(dcd_filename):
        l = dcd_filename[:dcd_filename.find(f"_energy_in_vacuum")].split('_')
        lam = l[-3]
        return float(lam)

    # calculate free energy
    def validate():
        'return free energy in kT'
        
        #return torch.tensor([5.0], device=device)
        if flipped:
            deltaF = fec.compute_free_energy_difference() * -1.
        else:
            deltaF = fec.compute_free_energy_difference()
        return deltaF

    # return the experimental value
    def experimental_value():
        e_in_kT = (exp_results[name]['energy'] * unit.kilocalorie_per_mole)/kT
        return torch.tensor([e_in_kT], device=device)

    if name in exclude_set_ANI + mols_with_charge:
        raise RuntimeError(f"{name} is part of the list of excluded molecules. Aborting")

    #######################
    exp_results = pickle.load(open('../data/exp_results.pickle', 'rb'))
    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']
    thinning = 50
    per_atom_stddev_threshold = 10.0 * unit.kilojoule_per_mole 
    latest_checkpoint = 'latest.pt'
    #######################

    print(f"Experimental free energy difference: {exp_results[name]['energy']} kcal/mol")

    #######################

    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles)
    tautomer = tautomers[0]
    tautomer.perform_tautomer_transformation()

    atoms = tautomer.hybrid_atoms
    top = tautomer.hybrid_topology

    # define the alchemical atoms
    alchemical_atoms = [tautomer.hybrid_hydrogen_idx_at_lambda_1, tautomer.hybrid_hydrogen_idx_at_lambda_0]

    # extract hydrogen donor idx and hydrogen idx for from_mol
    model = neutromeratio.ani.LinearAlchemicalSingleTopologyANI(alchemical_atoms=alchemical_atoms)

    model = model.to(device)
    torch.set_num_threads(1)

    # perform initial sampling
    energy_function = neutromeratio.ANI1_force_and_energy(
        model=model,
        atoms=atoms,
        mol=None,
        per_atom_thresh=10.4 * unit.kilojoule_per_mole,
        adventure_mode=True
    )

    # add restraints
    for r in tautomer.ligand_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)
    for r in tautomer.hybrid_ligand_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)

    # get steps inclusive endpoints
    # and lambda values in list
    dcds = glob(f"{data_path}/{name}/*.dcd")

    lambdas = []
    ani_trajs = []
    energies = []

    # read in all the frames from the trajectories
    for dcd_filename in dcds:
        lam = parse_lambda_from_dcd_filename(dcd_filename)
        lambdas.append(lam)
        traj = md.load_dcd(dcd_filename, top=top)[::thinning]
        print(f"Nr of frames in trajectory: {len(traj)}")
        ani_trajs.append(traj)
        f = open(f"{data_path}/{name}/{name}_lambda_{lam:0.4f}_energy_in_vacuum.csv", 'r')
        energies.append(np.array([float(e) for e in f][::thinning]))
        f.close()

    # calculate free energy in kT
    fec = FreeEnergyCalculator(ani_model=energy_function,
                                ani_trajs=ani_trajs,
                                potential_energy_trajs=energies,
                                lambdas=lambdas,
                                n_atoms=len(atoms),
                                max_snapshots_per_window=-1,
                                per_atom_thresh=per_atom_stddev_threshold)


    # defining neural networks
    nn = model.neural_networks
    aev_dim = model.aev_computer.aev_length
    # define which layer should be modified -- currently the last one
    layer = 6
    # take only a single network from the ensemble of 8
    weight_layers = []
    bias_layers = []
    for nn in model.neural_networks:
        weight_layers.extend(
            [
        {'params' : [nn.C[layer].weight], 'weight_decay': 0.000001},
        {'params' : [nn.H[layer].weight], 'weight_decay': 0.000001},
        {'params' : [nn.O[layer].weight], 'weight_decay': 0.000001},
        {'params' : [nn.N[layer].weight], 'weight_decay': 0.000001},
            ]
        )
        bias_layers.extend(
            [
        {'params' : [nn.C[layer].bias]},
        {'params' : [nn.H[layer].bias]},
        {'params' : [nn.O[layer].bias]},
        {'params': [nn.N[layer].bias]},
            ]
        )

    # set up minimizer for weights
    AdamW = torchani.optim.AdamW(weight_layers)

    # set up minimizer for bias
    SGD = torch.optim.SGD(bias_layers, lr=1e-3)

    AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)
    SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=100, threshold=0)

    # save checkpoint
    if os.path.isfile(latest_checkpoint):
        checkpoint = torch.load(latest_checkpoint)
        nn.load_state_dict(checkpoint['nn'])
        AdamW.load_state_dict(checkpoint['AdamW'])
        SGD.load_state_dict(checkpoint['SGD'])
        AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
        SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])


    print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
    max_epochs = 100
    early_stopping_learning_rate = 1.0E-5
    best_model_checkpoint = 'best.pt'

    for _ in tqdm(range(AdamW_scheduler.last_epoch + 1, max_epochs)):
        rmse = torch.sqrt((validate() - experimental_value())**2)
        print(rmse)
                
        # checkpoint
        if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
            torch.save(nn.state_dict(), best_model_checkpoint)

        # define the stepsize -- very conservative
        AdamW_scheduler.step(rmse/10)
        SGD_scheduler.step(rmse/10)
        loss = rmse

        AdamW.zero_grad()
        SGD.zero_grad()
        loss.backward()
        AdamW.step()
        SGD.step()

    torch.save({
        'nn': nn.state_dict(),
        'AdamW': AdamW.state_dict(),
        'SGD': SGD.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
        'SGD_scheduler': SGD_scheduler.state_dict(),
    }, latest_checkpoint)



if __name__ == '__main__':
    import neutromeratio
    import pickle
    import mdtraj as md
    from tqdm import tqdm
    
    # TODO: pkg_resources instead of filepath relative to execution directory
    exp_results = pickle.load(open('data/exp_results.pickle', 'rb'))
    # nr of steps
    #################
    n_steps = 20
    #################

    # specify the system you want to simulate
    name = 'molDWRow_298'  #Experimental free energy difference: 1.132369 kcal/mol
    # name = 'molDWRow_37'
    # name = 'molDWRow_45'
    # name = 'molDWRow_160'
    # name = 'molDWRow_590'
    if name in exclude_set_ANI + mols_with_charge:
        raise RuntimeError(f"{name} is part of the list of excluded molecules. Aborting")

    t1_smiles = exp_results[name]['t1-smiles']
    t2_smiles = exp_results[name]['t2-smiles']
    print(f"Experimental free energy difference: {exp_results[name]['energy']} kcal/mol")
    t_type, tautomers, flipped = neutromeratio.utils.generate_tautomer_class_stereobond_aware(name, t1_smiles, t2_smiles)
    tautomer = tautomers[0] # only considering ONE stereoisomer (the one deposited in the db)
    tautomer.perform_tautomer_transformation()

    # define the alchemical atoms
    alchemical_atoms = [tautomer.hybrid_hydrogen_idx_at_lambda_1, tautomer.hybrid_hydrogen_idx_at_lambda_0]

    # set the ANI model
    model = neutromeratio.ani.LinearAlchemicalSingleTopologyANI(alchemical_atoms=alchemical_atoms)
    model = model.to(device)
    torch.set_num_threads(1)

    # define energy function
    energy_function = neutromeratio.ANI1_force_and_energy(
            model=model,
            atoms=tautomer.hybrid_atoms,
            mol=None,)

    # add ligand bond restraints (for all lambda states)
    for r in tautomer.ligand_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)

    for r in tautomer.hybrid_ligand_restraints:
        energy_function.add_restraint_to_lambda_protocol(r)

    x0 = tautomer.hybrid_coords
    potential_energy_trajs = []
    ani_trajs = []
    lambdas = np.linspace(0, 1, 5)

    for lamb in tqdm(lambdas):
        # minimize coordinates with a given lambda value
        x0, e_history = energy_function.minimize(x0, maxiter=5000, lambda_value=lamb)
        # define energy function with a given lambda value
        energy_and_force = lambda x : energy_function.calculate_force(x, lamb)
        # define langevin object with a given energy function
        langevin = neutromeratio.LangevinDynamics(atoms=tautomer.hybrid_atoms,
                                        energy_and_force=energy_and_force)

        # sampling
        equilibrium_samples, energies, restraint_bias, stddev, ensemble_bias = langevin.run_dynamics(x0,
                                                                        n_steps=n_steps,
                                                                        stepsize=1.0*unit.femtosecond,
                                                                        progress_bar=False)

        potential_energy_trajs.append(np.array(energies))

        ani_trajs.append(md.Trajectory([x / unit.nanometer for x in equilibrium_samples], tautomer.hybrid_topology))

    # calculate free energy in kT
    fec = FreeEnergyCalculator(ani_model=energy_function,
                               ani_trajs=ani_trajs,
                               potential_energy_trajs=potential_energy_trajs,
                               lambdas=lambdas,
                               n_atoms=len(tautomer.hybrid_atoms),
                               max_snapshots_per_window=-1,
                               per_atom_thresh=1.0 * unit.kilojoule_per_mole)

    # BEWARE HERE: I change the sign of the result since if flipped is TRUE I have 
    # swapped tautomer 1 and 2 to mutate from the tautomer WITH the stereobond to the 
    # one without the stereobond
    if flipped:
        deltaF = fec.compute_free_energy_difference() * -1
    else:
        deltaF = fec.compute_free_energy_difference()
    print(f"Free energy difference {(deltaF.item() * kT).value_in_unit(unit.kilocalorie_per_mole)} kcal/mol")
    # let's say I had a loss function that wanted the free energy difference
    # estimate to be equal to 6:
    deltaF.backward()  # no errors or warnings

    params = list(energy_function.model.parameters())
    for p in params:
        print(p.grad)  # all None

    print('######################')
