import numpy as np
from typing import List
from yucca.functional.transforms.label_transforms import convert_labels_to_regions
from yucca.data.augmentation.transforms.YuccaTransform import YuccaTransform


class ConvertLabelsToRegions(YuccaTransform):
    def __init__(self, convert_labels_to_regions=False, label_key="label", regions: List[List[int]] = None):
        self.convert_labels_to_regions = convert_labels_to_regions
        self.label_key = label_key
        self.regions = regions
        if self.convert_labels_to_regions:
            assert self.regions is not None, "you cannot enable convert_labels_to_regions while not supplying any regions"

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __convert__(self, label: np.ndarray, regions: List[List[int]]):
        return convert_labels_to_regions(label, regions)

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.label_key].shape) == 5 or len(data_dict[self.label_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.label_key].shape}"
        if self.convert_labels_to_regions:
            data_dict[self.label_key] = self.__convert__(data_dict[self.label_key], self.regions)
        return data_dict
