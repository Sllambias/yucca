from yucca.image_processing.transforms.YuccaTransform import YuccaTransform
import numpy as np
from typing import Tuple


class MotionGhosting(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        p_per_sample=1,
        alpha=(0.85, 0.95),
        numReps=(2, 5),
        axes=(0, 3),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.alpha = alpha
        self.numReps = numReps
        self.axes = axes
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(alpha: Tuple[float], numReps: Tuple[float], axes: Tuple[float]) -> Tuple[float]:
        alpha = np.random.uniform(*alpha)
        numReps = np.random.randint(*numReps)
        axis = np.random.randint(*axes)
        return alpha, numReps, axis

    def __motionGhosting__(self, image, alpha, numReps, axis):
        img_min = image.min()
        img_max = image.max()
        m = min(0, img_min)
        image += abs(m)
        if len(image.shape) == 3:
            assert axis in [0, 1, 2], "Incorrect or no axis"

            h, w, d = image.shape

            image = np.fft.fftn(image, s=[h, w, d])

            if axis == 0:
                image[0:-1:numReps, :, :] = alpha * image[0:-1:numReps, :, :]
            elif axis == 1:
                image[:, 0:-1:numReps, :] = alpha * image[:, 0:-1:numReps, :]
            else:
                image[:, :, 0:-1:numReps] = alpha * image[:, :, 0:-1:numReps]

            image = abs(np.fft.ifftn(image, s=[h, w, d]))
        if len(image.shape) == 2:
            assert axis in [0, 1], "Incorrect or no axis"
            h, w = image.shape
            image = np.fft.fftn(image, s=[h, w])

            if axis == 0:
                image[0:-1:numReps, :] = alpha * image[0:-1:numReps, :]
            else:
                image[:, 0:-1:numReps] = alpha * image[:, 0:-1:numReps]
            image = abs(np.fft.ifftn(image, s=[h, w]))
        image -= abs(m)
        if self.clip_to_input_range:
            image = np.clip(image, a_min=img_min, a_max=img_max)
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
                    alpha, numReps, axis = self.get_params(self.alpha, self.numReps, self.axes)
                    data_dict[self.data_key][b, c] = self.__motionGhosting__(
                        data_dict[self.data_key][b, c], alpha, numReps, axis
                    )
        return data_dict
