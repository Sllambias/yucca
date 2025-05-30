from yucca.modules.data.augmentation.transforms.YuccaTransform import YuccaTransform
from yucca.functional.transforms import motion_ghosting, torch_motion_ghosting
import numpy as np
from typing import Tuple


class MotionGhosting(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        p_per_sample: float = 1.0,
        alpha=(0.85, 0.95),
        num_reps=(2, 5),
        axes=(0, 3),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.alpha = alpha
        self.num_reps = num_reps
        self.axes = axes
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(alpha: Tuple[float], num_reps: Tuple[float], axes: Tuple[float]) -> Tuple[float]:
        alpha = np.random.uniform(*alpha)
        num_reps = np.random.randint(*num_reps)
        axis = np.random.randint(*axes)
        return alpha, num_reps, axis

    def __motionGhosting__(self, image, alpha, num_reps, axis):
        image = motion_ghosting(
            image=image, alpha=alpha, num_reps=num_reps, axis=axis, clip_to_input_range=self.clip_to_input_range
        )
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
                    alpha, num_reps, axis = self.get_params(self.alpha, self.num_reps, self.axes)
                    data_dict[self.data_key][b, c] = self.__motionGhosting__(
                        data_dict[self.data_key][b, c], alpha, num_reps, axis
                    )
        return data_dict


class Torch_MotionGhosting(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        p_per_channel: float = 0.0,
        alpha=(0.85, 0.95),
        num_reps=(2, 5),
        axes=(0, 3),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_channel = p_per_channel
        self.alpha = alpha
        self.num_reps = num_reps
        self.axes = axes
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(alpha: Tuple[float], num_reps: Tuple[float], axes: Tuple[float]) -> Tuple[float]:
        alpha = np.random.uniform(*alpha)
        num_reps = np.random.randint(*num_reps)
        axis = np.random.randint(*axes)
        return alpha, num_reps, axis

    def __motionGhosting__(self, image, alpha, num_reps, axis):
        image = torch_motion_ghosting(
            image=image, alpha=alpha, num_reps=num_reps, axis=axis, clip_to_input_range=self.clip_to_input_range
        )
        return image

    def __call__(self, data_dict):
        for c in range(data_dict[self.data_key].shape[0]):
            if np.random.uniform() < self.p_per_channel:
                alpha, num_reps, axis = self.get_params(self.alpha, self.num_reps, self.axes)
                data_dict[self.data_key][c] = self.__motionGhosting__(data_dict[self.data_key][c], alpha, num_reps, axis)
        return data_dict
