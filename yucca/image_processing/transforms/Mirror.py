from yucca.image_processing.transforms.YuccaTransform import YuccaTransform
import numpy as np


class Mirror(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        label_key="label",
        p_per_sample=1,
        axes=(0, 1, 2),
        p_mirror_per_axis=0.33,
        skip_label=False,
    ):
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample
        self.p_mirror_per_axis = p_mirror_per_axis
        self.axes = axes
        self.skip_label = skip_label

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __mirror__(self, data_dict, axes):
        image = data_dict[self.data_key]
        label = data_dict.get(self.label_key)
        # Input will be [b, c, x, y, z] or [b, c, x, y]
        for b in range(image.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                if 0 in axes and np.random.uniform() < self.p_mirror_per_axis:
                    image[b, :, :] = image[b, :, ::-1]
                    if label is not None and not self.skip_label:
                        label[b, :, :] = label[b, :, ::-1]
                if 1 in axes and np.random.uniform() < self.p_mirror_per_axis:
                    image[b, :, :, :] = image[b, :, :, ::-1]
                    if label is not None and not self.skip_label:
                        label[b, :, :, :] = label[b, :, :, ::-1]
                if 2 in axes and np.random.uniform() < self.p_mirror_per_axis:
                    image[b, :, :, :, :] = image[b, :, :, :, ::-1]
                    if label is not None and not self.skip_label:
                        label[b, :, :, :, :] = label[b, :, :, :, ::-1]
        data_dict[self.data_key] = image
        if label is not None and not self.skip_label:
            data_dict[self.label_key] = label
        return data_dict

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"

        data_dict = self.__mirror__(
            data_dict,
            self.axes,
        )
        return data_dict
