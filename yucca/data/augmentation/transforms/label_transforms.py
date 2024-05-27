# %%
import numpy as np
from typing import List

from yucca.data.augmentation.transforms.YuccaTransform import YuccaTransform


class convert_segmentation_labels_to_regions(YuccaTransform):
    def __init__(self, convert_labels_to_regions=False, label_key="label"):
        self.convert_labels_to_regions = convert_labels_to_regions
        self.label_key = label_key

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __convert__(self, label: np.ndarray, regions: List[List[int]]):
        return convert_segmentation_to_regions(label, regions)

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.label_key].shape) == 5 or len(data_dict[self.label_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.label_key].shape}"
        if self.convert_labels_to_regions:
            data_dict[self.label_key] = self.__convert__(data_dict[self.label_key])
        return data_dict


def convert_segmentation_to_regions(label, regions):
    b, c, *shape = label.shape
    assert c == 1, "# Channels is not 1. Make sure the input to this function is a segmentation map of dims (b,c,h,w[,d])"
    region_canvas = np.zeros((b, len(regions) + 1, *shape))  # +1 to account for BG which is not a part of regions
    for channel, region in enumerate(regions):
        region_canvas[:, channel + 1] = np.isin(label[:, 0], region)
    region_canvas = region_canvas.astype(np.uint8)
    return region_canvas


import nibabel as nib
import matplotlib.pyplot as plt

path = "/Users/zcr545/Desktop/Projects/repos/yucca_data/raw_data/Task018_KITS23/labelsTr/KITS23_case_00002.nii.gz"
seg = nib.load(path).get_fdata()[np.newaxis, np.newaxis]
print(np.unique(seg, return_counts=True))

seg_converted = convert_segmentation_to_regions(seg, regions=[[1, 2, 3], [2, 3], [2]])
# %%
print(seg_converted[:, 1].sum())
print(seg_converted[:, 2].sum())
print(seg_converted[:, 3].sum())
# %%
np.isin(seg, [2, 3])
# %%
