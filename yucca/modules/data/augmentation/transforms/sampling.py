from yucca.modules.data.augmentation.transforms.YuccaTransform import YuccaTransform
from yucca.functional.transforms import downsample_label


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
        downsampled_labels = downsample_label(label, factors)
        return downsampled_labels

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        if self.deep_supervision:
            data_dict[self.label_key] = self.__downsample__(data_dict[self.label_key], self.factors)
        return data_dict
