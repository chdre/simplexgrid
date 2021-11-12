from noise_randomized import snoise2, randomize
import numpy as np


class SimplexGrid:
    def __init__(self, scale, threshold, l1, l2, n1, n2, **kwargs):
        self.grid1 = np.linspace(0, l1, n1)
        self.grid2 = np.linspace(0, l2, n2)
        self.n1 = n1
        self.n2 = n2
        self.noise_grid = np.zeros((n1, n2))
        self.scale = scale
        self.threshold = threshold
        self.kwargs = kwargs
        self.kwargs['repeatx'], self.kwargs['repeaty'] = l1 / \
            self.scale, l2 / self.scale

    def simplex(self):
        noise_grid = np.zeros((self.n1, self.n2))
        randomize(period=4096, seed=self.seed)

        noise_vals = np.array([snoise2(x / self.scale, y / self.scale, **self.kwargs)
                               for x in self.grid1 for y in self.grid2]).reshape(self.n1, self.n2)
        noise_grid += noise_vals > self.threshold

        return noise_grid

    def __call__(self, seed, base):
        self.seed = seed
        self.kwargs['base'] = base
        grid = self.simplex()

        return grid


class CreateMultipleSimplexGrids:
    def __init__(self, scales, thresholds, bases, octaves, l1, l2, n1, n2, N, seedgen, **kwargs):
        self.n1 = n1
        self.n2 = n2
        self.l1 = l1
        self.l2 = l2
        self.scales = scales
        self.thresholds = thresholds
        self.bases = bases
        self.octaves = octaves
        self.N = N
        self.kwargs = kwargs
        self.seedgen = seedgen

        self.dictionary = {'seed': [],
                           'scale': [],
                           'base': [],
                           'threshold': [],
                           'grid': [],
                           'octave': []}

    def __call__(self):
        rng = np.random.default_rng(42)
        n_bases = self.bases.shape[0]

        n_12 = self.n1 * self.n2

        for octave in self.octaves:
            for threshold in self.thresholds:
                for scale in self.scales:
                    bases_rng = rng.choice(self.bases, size=self.N, p=np.ones(
                        n_bases, dtype=np.float16) / n_bases)
                    simp_grid = SimplexGrid(scale=scale,
                                            threshold=threshold,
                                            l1=self.l1,
                                            l2=self.l2,
                                            n1=self.n1,
                                            n2=self.n2)

                    seeds = np.array([self.seedgen() for i in range(self.N)])

                    grids = np.array([simp_grid(seed=s, base=b)
                                     for s, b in zip(seeds, bases_rng)])

                    porosity = grids.sum(axis=(1, 2)) / n_12

                    # logical_and combined with inds = np.arange(..)[inds] below
                    # is >3x faster than argwhere
                    inds = np.logical_and(porosity > 0.1, porosity < 0.5)
                    if inds.sum() == 0:
                        break
                    # Turning boolean inds into actual indices
                    inds = np.arange(self.N)[inds]
                    # Creating an array of ones to use for octave, threshold and
                    # scale for storing in dictionary
                    n_inds = inds.shape[0]
                    tmp = np.ones(n_inds)

                    self.dictionary['seed'].append(seeds[inds])
                    self.dictionary['base'].append(bases_rng[inds])
                    self.dictionary['grid'].append(grids[inds])
                    self.dictionary['octave'].append(tmp * octave)
                    self.dictionary['threshold'].append(tmp * threshold)
                    self.dictionary['scale'].append(tmp * scale)

        return self.dictionary


class SeedGenerator:
    def __init__(self, start, step):
        self.start = start
        self.n = start
        self.step = step

        if not isinstance(start, int):
            raise ValueError('Start must be of type int')
        elif not isinstance(step, int):
            raise ValueError('Step must be of type int')

    def __iter__(self):
        self.n = self.start
        return self

    def __next__(self):
        self.n += 1
        return self.n

    def __str__(self):
        return f'Last seed: {self.n - 1}'

    def __call__(self):
        tmp = self.n
        self.n += 1
        return tmp
