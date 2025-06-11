from yucca.modules.data.augmentation.transforms.YuccaTransform import YuccaTransform
import numpy as np
from typing import Tuple
from yucca.functional.transforms import blur
from yucca.functional.transforms.torch.blur import torch_blur


class Blur(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        p_per_sample: float = 1.0,
        p_per_channel: float = 0.5,
        sigma=(0.5, 1.0),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.sigma = sigma
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(sigma: Tuple[float]):
        sigma = np.random.uniform(*sigma)
        return sigma

    def __blur__(self, image, sigma):
        for c in range(image.shape[0]):
            if np.random.uniform() < self.p_per_channel:
                image[c] = blur(image[c], sigma, clip_to_input_range=self.clip_to_input_range)
        return image

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"

        for b in range(data_dict[self.data_key].shape[0]):
            if np.random.uniform() < self.p_per_sample:
                sigma = self.get_params(self.sigma)
                data_dict[self.data_key][b] = self.__blur__(data_dict[self.data_key][b], sigma)
        return data_dict


class Torch_Blur(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        p_per_channel: float = 0.0,
        sigma=(0.5, 1.0),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_channel = p_per_channel
        self.sigma = sigma
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(sigma: Tuple[float]):
        sigma = np.random.uniform(*sigma)
        return sigma

    def __blur__(self, image, sigma):
        image = torch_blur(image, sigma, clip_to_input_range=self.clip_to_input_range)
        return image

    def __call__(self, data_dict):
        for c in range(data_dict[self.data_key].shape[0]):
            if np.random.uniform() < self.p_per_channel:
                sigma = self.get_params(self.sigma)
                data_dict[self.data_key][c] = self.__blur__(data_dict[self.data_key][c], sigma)
        return data_dict
