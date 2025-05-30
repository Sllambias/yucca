from yucca.modules.data.augmentation.transforms.YuccaTransform import YuccaTransform
from yucca.functional.transforms import gibbs_ringing, torch_gibbs_ringing
import numpy as np


class GibbsRinging(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        p_per_sample: float = 1.0,
        cut_freq=(96, 129),
        axes=(0, 3),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.cut_freq = cut_freq
        self.axes = axes
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(cut_freq, axes):
        cut_freq = np.random.randint(*cut_freq)
        axis = np.random.randint(*axes)
        return cut_freq, axis

    def __gibbsRinging__(self, image, num_sample, axis):
        image = gibbs_ringing(image, num_sample=num_sample, axis=axis, clip_to_input_range=self.clip_to_input_range)
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
                    cut_freq, axis = self.get_params(self.cut_freq, self.axes)
                    data_dict[self.data_key][b, c] = self.__gibbsRinging__(data_dict[self.data_key][b, c], cut_freq, axis)
        return data_dict


class Torch_GibbsRinging(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        p_per_channel: float = 0.0,
        cut_freq=(96, 129),
        axes=(0, 3),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_channel = p_per_channel
        self.cut_freq = cut_freq
        self.axes = axes
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(cut_freq, axes):
        cut_freq = np.random.randint(*cut_freq)
        axis = np.random.randint(*axes)
        return cut_freq, axis

    def __gibbsRinging__(self, image, num_sample, axis):
        image = torch_gibbs_ringing(image, num_sample=num_sample, axes=[axis], clip_to_input_range=self.clip_to_input_range)
        return image

    def __call__(self, data_dict):
        for c in range(data_dict[self.data_key].shape[0]):
            if np.random.uniform() < self.p_per_channel:
                cut_freq, axis = self.get_params(self.cut_freq, self.axes)
                data_dict[self.data_key][c] = self.__gibbsRinging__(data_dict[self.data_key][c], cut_freq, axis)
        return data_dict
