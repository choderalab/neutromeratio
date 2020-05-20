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
import pkg_resources
import memory_profiler

logger = logging.getLogger(__name__)

class FreeEnergyCalculator():
    def __init__(self,
                 ani_model: ANI1_force_and_energy,
                 md_trajs: list,
                 potential_energy_trajs: list,
                 lambdas: list,
                 n_atoms: int,
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
        """
        K = len(lambdas)
        assert (len(md_trajs) == K)
        assert (len(potential_energy_trajs) == K)
        self.ani_model = ani_model
        self.potential_energy_trajs = potential_energy_trajs  # for detecting equilibrium
        self.lambdas = lambdas
        self.n_atoms = n_atoms

        ani_trajs = {}
        for lam, traj, potential_energy in zip(self.lambdas, md_trajs, self.potential_energy_trajs):
            # detect equilibrium
            equil, g = detectEquilibration(np.array([e/kT for e in potential_energy]))[:2]
            # thinn snapshots and return max_snapshots_per_window confs
            quarter_traj_limit = int(len(traj) / 4)
            snapshots = traj[min(quarter_traj_limit, equil):].xyz * unit.nanometer
            further_thinning = max(int(len(snapshots) / max_snapshots_per_window), 1)
            snapshots = snapshots[::further_thinning][:max_snapshots_per_window]
            ani_trajs[lam] = snapshots
        
        del(md_trajs)
        snapshots = []
        N_k = []
        for lam in sorted(self.lambdas):
            print(f"lamb: {lam}")
            N_k.append(len(ani_trajs[lam]))
            snapshots.extend(ani_trajs[lam])
            logger.info(f"Snapshots per lambda {lam}: {len(ani_trajs[lam])}")

        assert (len(snapshots) > 20)
        
        coordinates = [sample / unit.angstrom for sample in snapshots] * unit.angstrom

        logger.debug(f"len(coordinates): {len(coordinates)}")
        logger.debug(f"coordinates: {coordinates[:5]}")

        # end-point energies
        lambda0_e = self.ani_model.calculate_energy(coordinates, lambda_value=0.).energy_tensor      
        lambda1_e = self.ani_model.calculate_energy(coordinates, lambda_value=1.).energy_tensor      

        logger.info(f"lambda0_e: {len(lambda0_e)}")
        logger.info(f"lambda0_e: {lambda0_e[:50]}")

        def get_mix(lambda0, lambda1, lam=0.0):
            return (1 - lam) * np.array(lambda0.detach()) + lam * np.array(lambda1.detach())

        logger.info('Nr of atoms: {}'.format(n_atoms))

        u_kn = np.stack(
            [get_mix(lambda0_e, lambda1_e, lam) for lam in sorted(self.lambdas)]
        )
        self.mbar = MBAR(u_kn, N_k)
        self.snapshots = snapshots


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

    def compute_perturbed_free_energies(self, u_ln):
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
        return u_ln

    def compute_free_energy_difference(self):
        u_ln = self.form_u_ln()
        f_k = self.compute_perturbed_free_energies(u_ln)
        return f_k[1] - f_k[0]


def get_free_energy_differences(fec_list:list)-> torch.Tensor:
    """
    Gets a list of fec instances and returns a torch.tensor with 
    the computed free energy differences.

    Arguments:
        fec_list {list[torch.tensor]} 

    Returns:
        torch.tensor -- calculated free energy in kT
    """
    calc = torch.tensor([0.0] * len(fec_list),
                                device=device, dtype=torch.float64)

    for idx, fec in enumerate(fec_list):
        #return torch.tensor([5.0], device=device)
        if fec.flipped:
            deltaF = fec.compute_free_energy_difference() * -1.
        else:
            deltaF = fec.compute_free_energy_difference()
        calc[idx] = deltaF
    print(calc)
    return calc

# return the experimental value
def get_experimental_values(names:list)-> torch.Tensor:
    """
    Returns the experimental free energy differen in solution for the tautomer pair

    Returns:
        [torch.Tensor] -- experimental free energy in kT
    """
    exp = torch.tensor([0.0] * len(names),
                                device = device, dtype = torch.float64)
    data = pkg_resources.resource_stream(__name__, "data/exp_results.pickle")
    exp_results = pickle.load(data)

    for idx, name in enumerate(names):
        e_in_kT = (exp_results[name]['energy'] * unit.kilocalorie_per_mole)/kT
        exp[idx] = e_in_kT
    return exp

def calculate_rmse(t1: torch.Tensor, t2: torch.Tensor):
    assert (t1.size() == t2.size())
    
    return torch.sqrt(torch.mean((t1 - t2)**2))

@profile
def tweak_parameters(names:list = ['SAMPLmol2'], data_path:str = "../data/", nr_of_nn:int = 8, max_epochs:int = 10):
    """
    Calculates the free energy of a staged free energy simulation, 
    tweaks the neural net parameter so that using reweighting the difference 
    between the experimental and calculated free energy is minimized.

    The function is set up to be called from the notebook or scripts folder.  

    Keyword Arguments:
        name {str} -- the name of the system (using the usual naming sheme) (default: {'SAMPLmol2'})
        data_path {str} -- should point to where the dcd files are located (default: {"../data/"})
        nr_of_nn {int} -- number of neural networks that should be tweeked, maximum 8  (default: {8})
    """

    #######################
    # some input parameters
    # 
    assert (int(nr_of_nn) <= 8)
    data = pkg_resources.resource_stream(__name__, "data/exp_results.pickle")
    print(f"data-filename: {data}")
    exp_results = pickle.load(data)
    latest_checkpoint = 'latest.pt'
    fec_list, model = neutromeratio.analysis.setup_mbar(names, data_path)


    # defining neural networks
    nn = model.neural_networks
    aev_dim = model.aev_computer.aev_length
    # define which layer should be modified -- currently the last one
    layer = 6
    # take each of the networks from the ensemble of 8
    weight_layers = []
    bias_layers = []
    for nn in model.neural_networks[:nr_of_nn]:
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
        {'params' : [nn.N[layer].bias]},
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
    early_stopping_learning_rate = 1.0E-5
    best_model_checkpoint = 'best.pt'

    h_exp_free_energy_difference = []
    for _ in tqdm(range(AdamW_scheduler.last_epoch + 1, max_epochs)):
        calc_free_energy_difference = get_free_energy_differences(fec_list)
        exp_free_energy_difference = get_experimental_values(names)
        rmse = calculate_rmse(calc_free_energy_difference, exp_free_energy_difference)
        logger.debug(f"RMSE: {rmse}")
        logger.debug(f"calc free energy difference: {exp_free_energy_difference}")
        h_exp_free_energy_difference.append(calc_free_energy_difference)  
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
    return h_exp_free_energy_difference