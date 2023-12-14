from yucca.image_processing.transforms.YuccaTransform import YuccaTransform
import numpy as np
from typing import Union, Callable


# Stolen from Batchgenerators to avoid import error caused by deprecated modules imported in
# Batchgenerators.
def augment_gamma(
    data_sample,
    gamma_range=(0.5, 2),
    invert_image=False,
    epsilon=1e-7,
    per_channel=False,
    retain_stats: Union[bool, Callable[[], bool]] = False,
):
    if invert_image:
        data_sample = -data_sample

    if not per_channel:
        retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
        if retain_stats_here:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats_here:
            data_sample = data_sample - data_sample.mean()
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
            data_sample = data_sample + mn
    else:
        for c in range(data_sample.shape[0]):
            retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
            if retain_stats_here:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats_here:
                data_sample[c] = data_sample[c] - data_sample[c].mean()
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
                data_sample[c] = data_sample[c] + mn
    if invert_image:
        data_sample = -data_sample
    return data_sample


class Gamma(YuccaTransform):
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

    def __init__(
        self,
        data_key="image",
        p_per_sample=1,
        p_invert_image=0.05,
        gamma_range=(0.5, 2.0),
        per_channel=True,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.gamma_range = gamma_range
        self.p_invert_image = p_invert_image
        self.per_channel = per_channel

    @staticmethod
    def get_params(p_invert_image):
        # No parameters to retrieve
        do_invert = False
        if np.random.uniform() < p_invert_image:
            do_invert = True
        return do_invert

    def __gamma__(self, imageVolume, gamma_range, invert_image, per_channel):
        return augment_gamma(imageVolume, gamma_range, invert_image, per_channel, retain_stats=False)

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"

        for b in range(data_dict[self.data_key].shape[0]):
            if np.random.uniform() < self.p_per_sample:
                do_invert = self.get_params(self.p_invert_image)
                data_dict[self.data_key][b] = self.__gamma__(
                    data_dict[self.data_key][b],
                    self.gamma_range,
                    do_invert,
                    per_channel=self.per_channel,
                )
        return data_dict
