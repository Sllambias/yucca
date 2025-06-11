from yucca.modules.data.augmentation.transforms.YuccaTransform import YuccaTransform
from yucca.functional.transforms import simulate_lowres, torch_simulate_lowres
import numpy as np


class SimulateLowres(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        p_per_sample: float = 1.0,
        p_per_channel: float = 0.5,
        p_per_axis: float = 0.33,
        zoom_range=(0.5, 1.0),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.p_per_axis = p_per_axis
        self.zoom_range = zoom_range
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(zoom_range, shape, p_per_axis):
        if isinstance(shape, (list, tuple)):
            shape = np.array(shape)
        zoom = np.random.uniform(*zoom_range)
        dim = len(shape)
        zoomed_shape = np.round(shape * zoom).astype(int)
        for i in range(dim):
            if np.random.uniform() < p_per_axis:
                shape[i] = zoomed_shape[i]
        return shape

    def __simulatelowres__(self, image, target_shape):
        image = simulate_lowres(image, target_shape=target_shape, clip_to_input_range=self.clip_to_input_range)
        return image

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"

        for b in range(data_dict[self.data_key].shape[0]):
            if np.random.uniform() < self.p_per_sample:
                for c in range(data_dict[self.data_key][b].shape[0]):
                    if np.random.uniform() < self.p_per_channel:
                        target_shape = self.get_params(
                            self.zoom_range,
                            data_dict[self.data_key][b, c].shape,
                            self.p_per_axis,
                        )
                        data_dict[self.data_key][b, c] = self.__simulatelowres__(data_dict[self.data_key][b, c], target_shape)
        return data_dict


class Torch_SimulateLowres(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        p_per_channel: float = 0.0,
        p_per_axis: float = 0.33,
        zoom_range=(0.5, 1.0),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_channel = p_per_channel
        self.p_per_axis = p_per_axis
        self.zoom_range = zoom_range
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(zoom_range, shape, p_per_axis):
        if isinstance(shape, (list, tuple)):
            shape = np.array(shape)
        zoom = np.random.uniform(*zoom_range)
        dim = len(shape)
        zoomed_shape = np.round(shape * zoom).astype(int)
        for i in range(dim):
            if np.random.uniform() < p_per_axis:
                shape[i] = zoomed_shape[i]
        return shape

    def __simulatelowres__(self, image, target_shape):
        image = torch_simulate_lowres(image, target_shape=target_shape, clip_to_input_range=self.clip_to_input_range)
        return image

    def __call__(self, data_dict):
        for c in range(data_dict[self.data_key].shape[0]):
            if np.random.uniform() < self.p_per_channel:
                target_shape = self.get_params(
                    self.zoom_range,
                    data_dict[self.data_key][c].shape,
                    self.p_per_axis,
                )
                data_dict[self.data_key][c] = self.__simulatelowres__(data_dict[self.data_key][c], target_shape)
        return data_dict
