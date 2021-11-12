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


class SeedGenerator:
    def __init__(self, start, step):
        self.start = start
        self.n = start
        self.step = step

        if start == 0:
            raise ValueError('Initial seed cannot be 0')
        elif not isinstance(start, int):
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
