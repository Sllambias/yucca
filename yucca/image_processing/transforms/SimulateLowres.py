from yucca.image_processing.transforms.YuccaTransform import YuccaTransform
import numpy as np
from skimage.transform import resize


class SimulateLowres(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        p_per_sample=1,
        p_per_channel=0.5,
        p_per_axis=0.33,
        zoom_range=(0.5, 1.0),
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.p_per_axis = p_per_axis
        self.zoom_range = zoom_range

    @staticmethod
    def get_params(zoom_range, shape, p_per_axis):
        # No parameters to retrieve
        if isinstance(shape, (list, tuple)):
            shape = np.array(shape)
        zoom = np.random.uniform(*zoom_range)
        dim = len(shape)
        zoomed_shape = np.round(shape * zoom).astype(int)
        for i in range(dim):
            if np.random.uniform() < p_per_axis:
                shape[i] = zoomed_shape[i]
        return shape

    def __simulatelowres__(self, imageVolume, target_shape):
        shape = imageVolume.shape
        downsampled = resize(
            imageVolume.astype(float),
            target_shape,
            order=0,
            mode="edge",
            anti_aliasing=False,
        )
        imageVolume = resize(downsampled, shape, order=3, mode="edge", anti_aliasing=False)
        return imageVolume

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
