import numpy as np
import torch
import os
import logging
from typing import Union, Literal, Optional
from batchgenerators.utilities.file_and_folder_operations import subfiles, load_pickle, isfile
from yucca.modules.data.augmentation.transforms.cropping_and_padding import CropPad
from yucca.modules.data.augmentation.transforms.formatting import NumpyToTorch
from yucca.modules.data.datasets.YuccaDataset import YuccaTrainDataset


class ClassificationDataset(YuccaTrainDataset):
    def __init__(
        self,
        samples: list,
        patch_size: list | tuple,
        keep_in_ram: Union[bool, None] = None,
        label_dtype: Optional[Union[int, float]] = torch.int32,
        task_type: str = "classification",
        composed_transforms=None,
        allow_missing_modalities=False,
        p_oversample_foreground=0.33,
    ):
        self.all_cases = samples
        self.composed_transforms = composed_transforms
        self.patch_size = patch_size
        self.label_dtype = label_dtype
        self.allow_missing_modalities = allow_missing_modalities
        self.already_loaded_cases = {}

        self.croppad = CropPad(patch_size=self.patch_size, label_key=None, p_oversample_foreground=p_oversample_foreground)
        self.to_torch = NumpyToTorch(label_dtype=self.label_dtype)

        self._keep_in_ram = keep_in_ram

    def __getitem__(self, idx):
        # remove extension if file splits include extensions
        case, _ = os.path.splitext(self.all_cases[idx])
        data = self.load_and_maybe_keep_volume(case)
        metadata = self.load_and_maybe_keep_pickle(case)

        if self.allow_missing_modalities:
            image, label = self.unpack_with_zeros(data)
        else:
            image, label = self.unpack(data)

        data_dict = {"file_path": case}
        data_dict.update({"image": image, "label": label})

        return self._transform(data_dict, metadata)

    def unpack(self, data):
        return data[0], data[-1][0]

    def unpack_with_zeros(self, data):
        assert data.dtype == "object", "allow missing modalities is true but dtype is not object"

        # First find the array with the largest array.
        # in classification this avoids setting the zero array to the 1d array with classes
        sizes = [i.size for i in data]
        idx_largest_array = np.where(sizes == np.max(sizes))[0][0]

        # replace missing modalities with zero-filed arrays
        for idx, i in enumerate(data):
            if i.size == 0:
                data[idx] = np.zeros(data[idx_largest_array].squeeze().shape)

        # unpack array into images and labels
        images = np.array([mod for mod in data[:-1]])
        label = data[-1:][0]

        return images, label
