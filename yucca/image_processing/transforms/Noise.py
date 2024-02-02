from yucca.image_processing.transforms.YuccaTransform import YuccaTransform
import numpy as np
from typing import Tuple


class AdditiveNoise(YuccaTransform):
    def __init__(self, data_key="image", p_per_sample=1, mean=(0.0, 0.0), sigma=(1e-3, 1e-4)):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.mean = mean
        self.sigma = sigma

    @staticmethod
    def get_params(mean: Tuple[float], sigma: Tuple[float]) -> Tuple[float]:
        mean = float(np.random.uniform(*mean))
        sigma = float(np.random.uniform(*sigma))
        return mean, sigma

    def __additiveNoise__(self, imageVolume, mean, sigma):
        # J = I+n
        gauss = np.random.normal(mean, sigma, imageVolume.shape)
        return imageVolume + gauss

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (c, x, y, z) or (c, x, y) and is: {data_dict[self.data_key].shape}"

        for b in range(data_dict[self.data_key].shape[0]):
            for c in range(data_dict[self.data_key][b].shape[0]):
                mean, sigma = self.get_params(self.mean, self.sigma)
                if np.random.uniform() < self.p_per_sample:
                    data_dict[self.data_key][b, c] = self.__additiveNoise__(data_dict[self.data_key][b, c], mean, sigma)
        return data_dict


class MultiplicativeNoise(YuccaTransform):
    """
    variables in DIKU_3D_augmentation_params:
        do_multiplicativeNoise
        multiplicativeNoise_p_per_sample
        multiplicativeNoise_mean
        multiplicativeNoise_sigma
    """

    def __init__(self, data_key="image", p_per_sample=1, mean=(0.0, 0.0), sigma=(1e-3, 1e-4)):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.mean = mean
        self.sigma = sigma

    @staticmethod
    def get_params(mean: Tuple[float], sigma: Tuple[float]) -> Tuple[float]:
        mean = float(np.random.uniform(*mean))
        sigma = float(np.random.uniform(*sigma))
        return mean, sigma

    def __multiplicativeNoise__(self, imageVolume, mean, sigma):
        # J = I + I*n
        gauss = np.random.normal(mean, sigma, imageVolume.shape)
        return imageVolume + imageVolume * gauss

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"

        for b in range(data_dict[self.data_key].shape[0]):
            for c in range(data_dict[self.data_key][b].shape[0]):
                if np.random.uniform() < self.p_per_sample:
                    mean, sigma = self.get_params(self.mean, self.sigma)
                    data_dict[self.data_key][b, c] = self.__multiplicativeNoise__(data_dict[self.data_key][b, c], mean, sigma)
        return data_dict
