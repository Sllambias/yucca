from yucca.image_processing.transforms.YuccaTransform import YuccaTransform
import numpy as np
from typing import Tuple
from scipy.ndimage import gaussian_filter


class Blur(YuccaTransform):
    """
    WRAPPER FOR NNUNET AUGMENT GAMMA: https://github.com/MIC-DKFZ/batchgenerators/blob/8822a08a7dbfa4986db014e6a74b040778164ca6/batchgenerators/augmentations/color_augmentations.py

    Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

    :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
    larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
    Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
    smaller than 1 and the other half with > 1
    :param invert_image: whether to invert the image before applying gamma augmentation
    :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
    the data will be transformed to match the mean and standard deviation before gamma augmentation. retain_stats
    can also be callable (signature retain_stats() -> bool)
    """

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
