from noise_randomized import snoise2, randomize
import numpy as np


class SimplexGrid:
    """Create a grid of stepwise Simplex noise.

    :param scales: Scale of the Simplex noise.
    :type scales: array_like
    :param thresholds: Threshold for Simplex values.
    :type thresholds: array_like
    :param bases: Base for Simplex values.
    :type bases: array_like
    :param octaves: Octaves for Simplex values.
    :type octaves: array_like
    :param l1: Length of system in a direction.
    :type l1: float
    :param l2: Length of system in a direction.
    :type l2: float
    :param n1: Number of grid cells in direction corresponding to length l1.
    :type n1: int
    :param n2: Number of grid cells in direction corresponding to length l2.
    :type n2: int
    """

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
        """Creates a Simplex grid.

        :returns noise_grid: Simplex grid
        :rtype noise_grid: np.ndarray
        """
        noise_grid = np.zeros((self.n1, self.n2))
        randomize(period=4096, seed=self.seed)

        noise_vals = np.array([
            snoise2(x / self.scale, y / self.scale, **self.kwargs)
            for x in self.grid1 for y in self.grid2
        ]).reshape(self.n1, self.n2)
        noise_grid += noise_vals > self.threshold

        return noise_grid

    def __call__(self, seed, base=0.0):
        """Returns the Simplex grid.

        :param seed: Seed for randomization
        :type seed: int
        :param base: Base for shifting noise coordinates. Defaults to zero, i.e.
                     no shift.
        :type base: float
        :returns grid: Simplex noise grid
        :rtype grid: np.ndarray
        """
        self.seed = seed
        self.kwargs['base'] = base
        grid = self.simplex()

        return grid


class Dictionary:
    """Creates a dictionary for keeping track of parameters.

    :param params: Variable name of parameters.
    :type params: list
    """

    def __init__(self, params):
        self.dictionary = {}
        for p in params:
            self.dictionary[p] = []

    def extend(self, **kwargs):
        for key, val in kwargs.items():
            if key in self.dictionary.keys():
                self.dictionary[key].extend(val)

    def __str__(self):
        return str(self.dictionary)

    def __call__(self):
        return self.dictionary

    def save(self, path, filename='dictionary'):
        """Saves the dictionary by pickle.

        :param path: Path to save dictionary.
        :type path: Pathlib.PosixPath
        :param filename: Filename of dictionary. Defaults to 'dictionary'.
        :type filename: str
        """
        with open(path, 'rb') as f:
            pickle.dump(path / filename, self.dictionary)


class CreateMultipleSimplexGrids(Dictionary):
    """Create multiple stepwise Simplex grids and store the parameters used in a
    dictionary.

    :param scales: Scale of the Simplex noise.
    :type scales: array_like
    :param thresholds: Threshold for Simplex values.
    :type thresholds: array_like
    :param bases: Base for Simplex values.
    :type bases: array_like
    :param octaves: Octaves for Simplex values.
    :type octaves: array_like
    :param l1: Length of system in a direction.
    :type l1: float
    :param l2: Length of system in a direction.
    :type l2: float
    :param n1: Number of grid cells in direction corresponding to length l1.
    :type n1: int
    :param n2: Number of grid cells in direction corresponding to length l2.
    :type n2: int
    :param N: Number of samples for each combination of scale, threshold, base
              and octave.
    :type N: int
    :param seedgen: Seed generator.
    :type seedgen: iterator
    :param criterion: Some criterion for which.
    :type criterion: function'
    """

    def __init__(self, scales, thresholds, octaves, l1, l2, n1, n2, N,
                 seedgen, bases=None, criterion=None, **kwargs):
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
        self.criterion = criterion

        self.x = np.arange(0, self.n1, 1)
        self.y = np.arange(0, self.n2, 1)
        self.px = np.ones(self.x.shape[0], dtype=np.int16) / self.x.shape[0]
        self.py = np.ones(self.y.shape[0], dtype=np.int16) / self.y.shape[0]

        # Create the parameters for the dictionary to only have parameters that
        # are not None.
        possible_params = ['scales', 'thresholds', 'octaves', 'bases']
        _locals = locals()
        params = [param for param in possible_params if _locals[param] is not None]
        # Extending the list
        params.extend(['seed', 'grid'])

        self.dictionary = Dictionary(params)

    def __call__(self):
        """Creates multiple Simplex grids and stores in dictionary.

        :returns dictionary: Dictionary containing Simplex parameters for a
                             specific grid.
        :rtype dictionary: dict
        """
        if self.bases is not None:
            rng = np.random.default_rng(42)
            n_bases = self.bases.shape[0]
        else:
            bases = self.bases

        for octave in self.octaves:
            for threshold in self.thresholds:
                for scale in self.scales:
                    simp_grid = SimplexGrid(scale=scale,
                                            threshold=threshold,
                                            l1=self.l1,
                                            l2=self.l2,
                                            n1=self.n1,
                                            n2=self.n2)

                    seeds = np.array([self.seedgen()
                                     for i in range(self.N)], dtype=int)
                    if self.bases is not None:
                        bases = rng.choice(self.bases, size=self.N, p=np.ones(
                            n_bases, dtype=np.float16) / n_bases)
                        grids = np.array([simp_grid(seed=s, base=b).astype(np.int8)
                                         for s, b in zip(seeds, bases)])
                    else:
                        grids = np.array([simp_grid(seed=s).astype(np.int8)
                                         for s in seeds])

                    # Rolling arrays
                    roll = np.array([(rng.choice(self.x, size=1, p=self.px)[0],
                                      rng.choice(self.y, size=1, p=self.py)[0])
                                     for i in range(self.N)])
                    grids = np.array([np.roll(grids[i], roll[i], (0, 1))
                                     for i in range(self.N)])

                    tmp = np.ones(self.N)
                    if self.criterion is not None:
                        inds = self.criterion(grids)
                        if inds.shape[0] > 0:
                            seeds = seeds[inds]
                            grids = grids[inds]
                            tmp = tmp[inds]
                            bases = bases[inds]

                            self.dictionary.extend(
                                seed=seeds,
                                base=bases,
                                octave=tmp * octave,
                                threshold=tmp * threshold,
                                scale=tmp * scale,
                                grid=grids
                            )
                    else:
                        self.dictionary.extend(
                            seed=seeds,
                            base=bases,
                            octave=tmp * octave,
                            threshold=tmp * threshold,
                            scale=tmp * scale,
                            grid=grids
                        )
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
