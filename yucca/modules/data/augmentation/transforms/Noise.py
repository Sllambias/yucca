from yucca.modules.data.augmentation.transforms.YuccaTransform import YuccaTransform
from yucca.functional.transforms import additive_noise, multiplicative_noise, torch_additive_noise, torch_multiplicative_noise
import numpy as np
from typing import Tuple


class AdditiveNoise(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        p_per_sample: float = 1.0,
        mean=(0.0, 0.0),
        sigma=(1e-3, 1e-4),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
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

        for b in range(data_dict[self.data_key].shape[0]):
            for c in range(data_dict[self.data_key][b].shape[0]):
                mean, sigma = self.get_params(self.mean, self.sigma)
                if np.random.uniform() < self.p_per_sample:
                    data_dict[self.data_key][b, c] = self.__additiveNoise__(data_dict[self.data_key][b, c], mean, sigma)
        return data_dict


class MultiplicativeNoise(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        p_per_sample: float = 1.0,
        mean=(0.0, 0.0),
        sigma=(1e-3, 1e-4),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
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

        for b in range(data_dict[self.data_key].shape[0]):
            for c in range(data_dict[self.data_key][b].shape[0]):
                if np.random.uniform() < self.p_per_sample:
                    mean, sigma = self.get_params(self.mean, self.sigma)
                    data_dict[self.data_key][b, c] = self.__multiplicativeNoise__(data_dict[self.data_key][b, c], mean, sigma)
        return data_dict


class Torch_AdditiveNoise(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        p_per_channel: float = 0.0,
        mean=(0.0, 0.0),
        sigma=(1e-3, 1e-4),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
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
        image = torch_additive_noise(image=image, mean=mean, sigma=sigma, clip_to_input_range=self.clip_to_input_range)
        return image

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        for c in range(data_dict[self.data_key].shape[0]):
            if np.random.uniform() < self.p_per_channel:
                mean, sigma = self.get_params(self.mean, self.sigma)
                data_dict[self.data_key][c] = self.__additiveNoise__(data_dict[self.data_key][c], mean, sigma)
        return data_dict


class Torch_MultiplicativeNoise(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        p_per_channel: float = 0.0,
        mean=(0.0, 0.0),
        sigma=(1e-3, 1e-4),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
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
        image = torch_multiplicative_noise(image=image, mean=mean, sigma=sigma, clip_to_input_range=self.clip_to_input_range)
        return image

    def __call__(self, data_dict):
        for c in range(data_dict[self.data_key].shape[0]):
            if np.random.uniform() < self.p_per_channel:
                mean, sigma = self.get_params(self.mean, self.sigma)
                data_dict[self.data_key][c] = self.__multiplicativeNoise__(data_dict[self.data_key][c], mean, sigma)
        return data_dict
