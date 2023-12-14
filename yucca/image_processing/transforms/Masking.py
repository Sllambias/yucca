import numpy as np
from yucca.image_processing.transforms.YuccaTransform import YuccaTransform


class Masking(YuccaTransform):
    """
    CURRENTLY NOT IMPLEMENTED
    """

    def __init__(self, mask=False, data_key="image", mask_ratio: tuple | float = 0.25):
        self.mask = mask
        self.data_key = data_key
        self.mask_ratio = mask_ratio

    @staticmethod
    def get_params(shape, ratio, start_idx):
        pass

    def __mask__(self, image, label, crop_start_idx):
        pass

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        if self.mask:
            raise NotImplementedError("Masking is not implemented yet. It should not be enabled")
        return data_dict
