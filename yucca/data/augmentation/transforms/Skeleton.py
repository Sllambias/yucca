from yucca.data.augmentation.transforms.YuccaTransform import YuccaTransform
from typing import Tuple
from yucca.functional.transforms import skeleton

import numpy as np


class SkeletonTransform(YuccaTransform):
    def __init__(self, skeleton=False, label_key="label", do_tube=True):
        self.label_key = label_key
        self.do_tube = do_tube


    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __skeleton__(self, array):
        array_new = array.copy()
        # print(array_new.shape[0])
        for c in range(0, array_new.shape[0]):
            array_new[c, :, :] = skeleton(array_new[c, :, :])
        return array_new

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):

        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.label_key].shape) == 5 or len(data_dict[self.label_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (c, x, y, z) or (c, x, y) and is: {data_dict[self.label_key].shape}"
        if self.__skeleton__:
            data_dict["skel"] = self.__skeleton__(data_dict[self.label_key])
        return data_dict
