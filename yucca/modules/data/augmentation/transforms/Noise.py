from yucca.modules.data.augmentation.transforms.YuccaTransform import YuccaTransform
from yucca.functional.transforms import additive_noise, multiplicative_noise
import numpy as np
from typing import Tuple


class AdditiveNoise(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        p_per_sample: float = 1.0,
        p_per_channel=1.0,
        mean=(0.0, 0.0),
        sigma=(1e-3, 1e-4),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.mean = mean
        self.sigma = sigma
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(mean: Tuple[float], sigma: Tuple[float]) -> Tuple[float]:
        mean = float(np.random.uniform(*mean))
        sigma = float(np.random.uniform(*sigma))
        return mean, sigma

    def __additiveNoise__(self, image, mean, sigma):
        image = additive_noise(image=image, mean=mean, sigma=sigma, clip_to_input_range=self.clip_to_input_range)
        return image

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (c, x, y, z) or (c, x, y) and is: {data_dict[self.data_key].shape}"

        if not isinstance(self.p_per_channel, (list, tuple)):
            self.p_per_channel = [self.p_per_channel for _ in data_dict[self.data_key].shape[1]]

        for b in range(data_dict[self.data_key].shape[0]):
            if np.random.uniform() < self.p_per_sample:
                for c in range(data_dict[self.data_key][b].shape[0]):
                    if np.random.uniform() < self.p_per_channel[c]:
                        mean, sigma = self.get_params(self.mean, self.sigma)
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

    def __init__(
        self,
        data_key="image",
        p_per_sample: float = 1.0,
        p_per_channel=1.0,
        mean=(0.0, 0.0),
        sigma=(1e-3, 1e-4),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.mean = mean
        self.sigma = sigma
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(mean: Tuple[float], sigma: Tuple[float]) -> Tuple[float]:
        mean = float(np.random.uniform(*mean))
        sigma = float(np.random.uniform(*sigma))
        return mean, sigma

    def __multiplicativeNoise__(self, image, mean, sigma):
        image = multiplicative_noise(image=image, mean=mean, sigma=sigma, clip_to_input_range=self.clip_to_input_range)
        return image

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"

        if not isinstance(self.p_per_channel, (list, tuple)):
            self.p_per_channel = [self.p_per_channel for _ in data_dict[self.data_key].shape[1]]

        for b in range(data_dict[self.data_key].shape[0]):
            if np.random.uniform() < self.p_per_sample:
                for c in range(data_dict[self.data_key][b].shape[0]):
                    if np.random.uniform() < self.p_per_channel[c]:
                        mean, sigma = self.get_params(self.mean, self.sigma)
                        data_dict[self.data_key][b, c] = self.__multiplicativeNoise__(
                            data_dict[self.data_key][b, c], mean, sigma
                        )
        return data_dict
