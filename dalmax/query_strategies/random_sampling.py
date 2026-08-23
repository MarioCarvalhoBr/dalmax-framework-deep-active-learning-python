import numpy as np

from .base import Strategy


class RandomSampling(Strategy):
    def __init__(self, dataset, net, logger):
        super().__init__(dataset, net, logger)

    def query(self, n):
        print(f"Initializing the DAL strategy with RandomSampling query {n} samples")
        samples = np.random.choice(np.where(self.dataset.labeled_idxs==0)[0], n, replace=False)
        print(f"Samples selected: {samples}")
        return samples
