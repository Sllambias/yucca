from yucca.modules.data.augmentation.transforms.YuccaTransform import YuccaTransform
import numpy as np
from yucca.functional.transforms import bias_field


class BiasField(YuccaTransform):

    def __init__(
        self,
        data_key="image",
        p_per_sample: float = 1.0,
        p_per_channel=1.0,
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __biasField__(self, image):
        image = bias_field(image, clip_to_input_range=self.clip_to_input_range)
        return image

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape. \nShould be (b, c, x, y, z) or (b, c, x, y) and is:\
                {data_dict[self.data_key].shape}"

        if not isinstance(self.p_per_channel, (list, tuple)):
            self.p_per_channel = [self.p_per_channel for _ in data_dict[self.data_key].shape[1]]

        for b in range(data_dict[self.data_key].shape[0]):
            if np.random.uniform() < self.p_per_sample:
                for c in range(data_dict[self.data_key][b].shape[0]):
                    if np.random.uniform() < self.p_per_channel[c]:
                        data_dict[self.data_key][b, c] = self.__biasField__(data_dict[self.data_key][b, c])
        return data_dict
