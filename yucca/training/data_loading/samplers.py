import torch
import numpy as np


class RandomBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int = None):
        assert len(dataset) > 0
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        yield np.random.choice(len(self.dataset), self.batch_size)


class InfiniteRandomSampler(torch.utils.data.Sampler):
    """Return random indices from [0-n) infinitely.

    Arguments:
        dset_size (int): Size of the dataset to sample.
    """

    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dset_size = len(dataset)

    def __iter__(self):
        # Create a random number generator (optional, makes the sampling independent of the base RNG)
        rng = torch.Generator()
        seed = torch.empty((), dtype=torch.int64).random_().item()
        rng.manual_seed(seed)
        return _infinite_generator(self.dset_size, rng)

    def __len__(self):
        return float("inf")


def _infinite_generator(n, rng):
    """Inifinitely returns a number in [0, n)."""
    while True:
        yield from torch.randperm(n, generator=rng).tolist()
