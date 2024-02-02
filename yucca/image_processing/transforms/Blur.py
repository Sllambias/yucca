from yucca.image_processing.transforms.YuccaTransform import YuccaTransform
import numpy as np
from typing import Tuple
from scipy.ndimage import gaussian_filter


class Blur(YuccaTransform):
    def __init__(self, data_key="image", p_per_sample=1, p_per_channel=0.5, sigma=(0.5, 1.0)):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.sigma = sigma

    @staticmethod
    def get_params(sigma: Tuple[float]):
        sigma = np.random.uniform(*sigma)
        return sigma

    def __blur__(self, imageVolume, sigma):
        for c in range(imageVolume.shape[0]):
            if np.random.uniform() < self.p_per_channel:
                imageVolume[c] = gaussian_filter(imageVolume[c], sigma, order=0)
        return imageVolume

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
