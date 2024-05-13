import torch
import numpy as np
import logging


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


class InfiniteRandomSampler_with_oversampling(torch.utils.data.Sampler):
    """Return random indices from [0-n) infinitely.

    Arguments:
        dset_size (int): Size of the dataset to sample.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        positive_indexes: list,
        p_oversample_foreground: float,
    ):
        logging.warn(
            "You are using the sampler: InfiniteRandomSampler_OversampleForeground. "
            "This should not be used with shuffled datasets as the positive indexes "
            "Will no longer correspond with the correct dataset indexes. "
        )
        self.dset_size = len(dataset)
        self.pos_size = len(positive_indexes)
        self.pos_indexes = positive_indexes
        self.p_oversample_foreground = p_oversample_foreground

    def __iter__(self):
        seed = torch.empty((), dtype=torch.int64).random_().item()
        rng = np.random.default_rng(seed=seed)
        return _infinite_generator_with_oversampling(
            self.dset_size, pos_indexes=self.pos_indexes, rng=rng, p_oversample_foreground=self.p_oversample_foreground
        )

    def __len__(self):
        return float("inf")


def _infinite_generator(n, rng):
    """Inifinitely returns a number in [0, n)."""
    while True:
        yield from torch.randperm(n, generator=rng).tolist()


def _infinite_generator_with_oversampling(n_total, pos_indexes, rng, p_oversample_foreground):
    while True:
        if np.random.uniform() >= p_oversample_foreground:
            yield from rng.choice(n_total, 1)
        else:
            yield from rng.choice(pos_indexes, 1)
