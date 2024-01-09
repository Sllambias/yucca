from yucca.image_processing.transforms.YuccaTransform import YuccaTransform
import numpy as np
from skimage.transform import resize


class DownsampleSegForDS(YuccaTransform):
    """ """

    def __init__(self, deep_supervision: bool = False, label_key="label", factors=(1, 0.5, 0.25, 0.125, 0.0625)):
        self.deep_supervision = deep_supervision
        self.label_key = label_key
        self.factors = factors

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __downsample__(self, label, factors):
        orig_type = label.dtype
        orig_shape = label.shape
        downsampled_labels = []
        for factor in factors:
            target_shape = np.array(orig_shape).astype(int)
            for i in range(2, len(orig_shape)):
                target_shape[i] *= factor
            if np.all(target_shape == orig_shape):
                downsampled_labels.append(label)
            else:
                canvas = np.zeros(target_shape)
                for b in range(label.shape[0]):
                    for c in range(label[b].shape[0]):
                        canvas[b, c] = resize(
                            label[b, c].astype(float),
                            target_shape[2:],
                            0,
                            mode="edge",
                            clip=True,
                            anti_aliasing=False,
                        ).astype(orig_type)
                downsampled_labels.append(canvas)
        return downsampled_labels

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        if self.deep_supervision:
            data_dict[self.label_key] = self.__downsample__(data_dict[self.label_key], self.factors)
        return data_dict
