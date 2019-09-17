import numpy as np
from .constants import speed_unit, distance_unit, kB
from simtk import unit
from tqdm import tqdm
from .ani import ANI1_force_and_energy
import mdtraj as md


class LangevinDynamics(object):

    def __init__(self, atoms:str, temperature:int, force:ANI1_force_and_energy):
        self.force = force
        self.temperature = temperature
        self.atoms = atoms

    def run_dynamics(self, 
                    x0:np.ndarray,
                    lambda_value:float = 0.0,
                    n_steps:int = 100,
                    stepsize:unit.quantity.Quantity = 1.0*unit.femtosecond,
                    collision_rate:unit.quantity.Quantity = 10/unit.picoseconds,
                    progress_bar:bool = False
            ):
        """Unadjusted Langevin dynamics.

        Parameters
        ----------
        x0 : array of floats, unit'd (distance unit)
            initial configuration
        force : callable, accepts a unit'd array and returns a unit'd array
            assumes input is in units of distance
            output is in units of energy / distance
        lambda_value: float, between 0 and 1
            position in the lambda protocol; from 0 to 1  
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
        assert(float(lambda_value) <= 1.0 and float(lambda_value) >= 0.0)


        # generate mass arrays
        mass_dict_in_daltons = {'H': 1.0, 'C': 12.0, 'N': 14.0, 'O': 16.0}
        masses = np.array([mass_dict_in_daltons[a] for a in self.atoms]) * unit.daltons
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
        F, E = self.force.calculate_force(x, lambda_value)
        # energy is saved as a list
        energy = [E]

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
            F, E = self.force.calculate_force(x, lambda_value)
            energy.append(E)
            # v
            v += (stepsize * 0.5) * F / masses[:,None]

            norm_F = np.linalg.norm(F)
            # report gradient norm
            if progress_bar:
                trange.set_postfix({'|force|': norm_F})
            # check positions and forces are finite
            if (not np.isfinite(x).all()) or (not np.isfinite(norm_F)):
                print("Numerical instability encountered!")
                return traj, energy
            traj.append(x)
        return traj, energy


        
        

def use_precalculated_md_and_performe_mc(top:str,
                                        trajs:list,
                                        hydrogen_movers:list,
                                        mc_every_nth_frame:int):

    """
    Iterates over a trajectory and performs MC moves.
    The hydrogen_movers specify a list of MC_mover objects that should be used on the same coordinate set. 
    Parameters
    ----------
    top : str
            file path to topology file
    trajs: list[str]
            list of file paths to traj files
    hydrogen_movers: list[MC_mover]
            all MC_movers specified in this list are subsequently applied to the same coordinate set
    mc_every_nth_frame: int
            performs MC every nth frame

    """
    topology = md.load(top).topology
    traj = md.load(trajs, top=topology)
    for x in traj[::mc_every_nth_frame]:
        coordinates = x.xyz[0] * unit.nanometer
        for hydrogen_mover in hydrogen_movers:
            # MC move
            new_coordinates, work = hydrogen_mover.perform_mc_move(coordinates)
            hydrogen_mover.proposed_coordinates.append(new_coordinates)
            hydrogen_mover.initial_coordinates.append(coordinates)
            hydrogen_mover.work_values.append(work)
            

class MonteCarloBarostat(object):

    def __init__(self, pbc_box_length:unit.Quantity, energy:ANI1_force_and_energy):

        assert(type(pbc_box_length) == unit.Quantity)
        self.current_volumn = pbc_box_length ** 3
        self.num_attempted = 0
        self.num_accepted = 0
        self.volume_scale = 0.01 * self.current_volumn
        self.energy_function = energy

    def update_volumn(self, x:unit.Quantity):
        raise(NotImplementedError('under construction!'))

        assert(type(x) == unit.Quantity)
        # TODO: x is not used
        print(self.energy_function.model.pbc)
        energy = self.energy_function.calculate_energy(x)
        current_volumn = self.current_volumn
        import random
        delta_volumn = current_volumn * 2 * (random.uniform(0, 1) - 0.5)
        # TODO: use numpy.random throughout -- slightly different conventions
        new_volumn = current_volumn + delta_volumn
        length_scale = (new_volumn/current_volumn) ** (1.0/3.0)
        # TODO: length_scale not used


def read_precalculated_md(top:str, trajs:list):
    """

    Parameters
    ----------
    top : str
            file path to topology file
    trajs: list[str]
            list of file paths to traj files

    Returns
    -------
    traj_in_mm : list of (n_atoms,3) numpy arrays with nanometer units attached?
    """
    topology = md.load(top).topology
    traj = md.load(trajs, top=topology)
    # TODO: update this a bit
    traj_in_nm = []
    for x in traj:
        coordinates = x.xyz[0] * unit.nanometer
        traj_in_nm.append(coordinates)
    return traj_in_nm
