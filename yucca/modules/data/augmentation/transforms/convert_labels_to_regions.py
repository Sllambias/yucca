import numpy as np
from typing import List
from yucca.functional.transforms.label_transforms import batch_convert_labels_to_regions, translate_region_labels
from yucca.modules.data.augmentation.transforms.YuccaTransform import YuccaTransform


class ConvertLabelsToRegions(YuccaTransform):
    def __init__(
        self,
        convert_labels_to_regions=False,
        label_key="label",
        labels: dict[str, str] = None,
        regions: dict[str, dict] = None,
    ):
        self.convert_labels_to_regions = convert_labels_to_regions
        self.label_key = label_key

        if self.convert_labels_to_regions:
            assert regions is not None, "you cannot enable convert_labels_to_regions while not supplying any regions"
            assert labels is not None, "you cannot enable convert_labels_to_regions while not supplying labels"
            # regions stores region labels in str format for readability,
            # but here we need the actual label integer from the label map
            # so we translate it to integers
            self.regions = translate_region_labels(regions, labels)

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __convert__(self, label: np.ndarray, regions: List[List[int]]):
        return batch_convert_labels_to_regions(label, regions)

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        if self.convert_labels_to_regions:
            assert (
                len(data_dict[self.label_key].shape) == 5 or len(data_dict[self.label_key].shape) == 4
            ), f"Incorrect data size or shape.\
                \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.label_key].shape}"
            data_dict[self.label_key] = self.__convert__(data_dict[self.label_key], self.regions)
        return data_dict
