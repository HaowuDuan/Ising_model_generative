import numpy as np
from joblib import Parallel, delayed

class Params:
    def __init__(self,L,beta, J, H, N_sample):
        self.L=L
        self.beta=beta
        self.J=J
        self.H=H
        self.N_burnin=100*(L)**2
        self.N_sample=N_sample
        self.N_separation=100*(L)**2


class IsingModelDataset:
    def __init__(self, Params):
        # params containing L, beta, J, H, N_burnin, N_sample, N_separation.
 
        self.params = Params
        self.lattices = [np.random.choice([-1, 1], size=(self.params.L, self.params.L)) for _ in self.params.beta]
        self.magnetizations = [self._compute_initial_magnetization(lattice) for lattice in self.lattices]
        self.energies = [self._compute_initial_energy(lattice) for lattice in self.lattices]
        self.observables = None
        self.configs = None

    def _compute_initial_magnetization(self, lattice):
        #Compute the initial magnetization of the lattice
        return np.sum(lattice)

    def _compute_initial_energy(self, lattice):
       # Compute the initial energy of the lattice
        e = 0
        for i in range(self.params.L):
            for j in range(self.params.L):
                e -= self.params.J * lattice[i, j] * (
                    lattice[(i + 1) % self.params.L, j] +
                    lattice[i, (j + 1) % self.params.L]
                )
                e += self.params.H * lattice[i, j]
        return e

    def _update_spin(self, i, j,lattice,beta):
       
        dE = -2 * (-self.params.J) * lattice[i, j] * (
                 lattice[(i + 1) % self.params.L, j] +
                 lattice[(i - 1) % self.params.L, j] +
                 lattice[i, (j + 1) % self.params.L] +
                 lattice[i, (j - 1) % self.params.L]
        )
        dE += 2 * self.params.H *lattice[i, j]

        if np.random.rand() < np.exp(-beta * dE):
            lattice[i, j] *= -1
            return dE, 2 * lattice[i, j]
        return 0, 0

    def _configs_for_single_beta(self,lattice,beta,energy,magnetization):
        lattice = lattice.copy()
        energy = energy
        magnetization = magnetization
        configs = []
        observables = np.zeros((self.params.N_sample, 2))

        # thermalization stage
        for _ in range(self.params.N_burnin):
            i, j = np.random.randint(0, self.params.L, size=2)
            dE, dm = self._update_spin(i, j,lattice,beta)
            energy += dE
            magnetization += dm

        # Measurement stage

        for k in range(self.params.N_sample):
            for _ in range(self.params.N_separation):
                i, j = np.random.randint(0, self.params.L, size=2)
                dE, dm = self._update_spin(i, j,lattice,beta)
                energy += dE
                magnetization += dm
            observables[k] = [magnetization, energy]
            configs.append(lattice.copy())
        
        return np.array(configs), observables   

    # get configs for all betas
    def get_configs(self):
        # Run generation for each beta in parallel
        results = Parallel(n_jobs=-1)(
            delayed(self._configs_for_single_beta)(
                 self.lattices[i],beta,self.energies[i], self.magnetizations[i]
            )
            for i, beta in enumerate(self.params.beta)
        )

        # Unpack results
        self.configs = np.array([res[0] for res in results])
        self.observables = np.array([res[1] for res in results])