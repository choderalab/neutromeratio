import neutromeratio

class BFGS(object):

    def __init__(self, energy_function:neutromeratio.ani.ANI1_force_and_energy, maxstep:float=0.04, coor=None):
        self.maxstep = maxstep
        self.calculate_energy = energy_function.calculate_energy 
        self.calculate_forces = energy_function.calculate_force
        self.coor = coor

    def initialize(self):
        self.H = None
        self.r0 = None
        self.f0 = None
 
    def step(self, f=None):
        
        atoms = self.atoms

        if f is None:
            f = self.calculate_forces(self.coor)

        self.update(self.coor, f, self.r0, self.f0)
        omega, V = eigh(self.H)
        dr = np.dot(V, np.dot(f, V) / np.fabs(omega)).reshape((-1, 3))
        steplengths = (dr**2).sum(1)**0.5
        dr = self.determine_step(dr, steplengths)
        atoms.set_positions(r + dr)
        self.r0 = r.flat.copy()
        self.f0 = f.copy()
        self.dump((self.H, self.r0, self.f0, self.maxstep))


    def update(self, r, f, r0, f0):
        if self.H is None:
            self.H = np.eye(3 * len(self.atoms)) * 70.0
            return
        dr = r - r0

        if np.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        df = f - f0
        a = np.dot(dr, df)
        dg = np.dot(self.H, dr)
        b = np.dot(dr, dg)
        self.H -= np.outer(df, df) / a + np.outer(dg, dg) / b
