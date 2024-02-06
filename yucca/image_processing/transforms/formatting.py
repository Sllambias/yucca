import numpy as np
import torch
from yucca.image_processing.transforms.YuccaTransform import YuccaTransform
from typing import Optional


class RemoveSegChannelAxis(YuccaTransform):
    def __init__(self, label_key="label", channel_to_remove=1):
        self.label_key = label_key
        self.channel = channel_to_remove

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __remove_channel__(self, image, channel):
        return np.squeeze(image, axis=channel)

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            data_dict[self.label_key].shape[self.channel] == 1
        ), f"Invalid operation: attempting to remove channel of size > 1.\
            \nTrying to squeeze channel: {self.channel} of array with shape: {data_dict[self.label_key].shape}"
        data_dict[self.label_key] = self.__remove_channel__(data_dict[self.label_key], self.channel)
        return data_dict


class NumpyToTorch(YuccaTransform):
    def __init__(self, data_key="image", label_key="label", label_dtype: Optional[torch.dtype] = None):
        self.data_key = data_key
        self.label_key = label_key
        self.label_dtype = label_dtype

    def get_params(self, label):
        if self.label_dtype is not None or label is None:  # Nothing to infer here.
            return
        if isinstance(label, list):  # We just want to look at the first array
            label = label[0]
        if isinstance(label, np.floating):
            self.label_dtype = torch.float32
        else:
            self.label_dtype = torch.int32

    def __convert__(self, datadict):
        data = torch.tensor(datadict[self.data_key], dtype=torch.float32)
        label = datadict.get(self.label_key)

        if label is None:
            return {self.data_key: data}

        if isinstance(label, list):
            label = [torch.tensor(i, dtype=self.label_dtype) for i in label]
        else:
            label = torch.tensor(label, dtype=self.label_dtype)
        return {self.data_key: data, self.label_key: label}

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        data_len = len(data_dict[self.data_key].shape)
        assert (
            data_len == 5 or data_len == 4 or data_len == 3  # (B, C, H, W, D)  # (C, H, W, D) or (B, C, H, W)  # (C, H, W)
        ), f"Incorrect data size or shape.\
            \nShould be (B, C, X, Y, Z) or (B, C, X, Y) or (C, X, Y, Z) or (C, X, Y) and is: {data_len}"
        self.get_params(data_dict.get(self.label_key))
        data_dict = self.__convert__(data_dict)
        return data_dict


class AddBatchDimension(YuccaTransform):
    def __init__(self, data_key="image", label_key="label"):
        self.data_key = data_key
        self.label_key = label_key

    @staticmethod
    def get_params():
        pass

    def __unsqueeze__(self, data, label):
        data = data[np.newaxis]
        if label is None:
            return data, label
        if isinstance(label, list):
            label = [s[np.newaxis] for s in label]
        else:
            label = label[np.newaxis]
        return data, label

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        data_dict[self.data_key], data_dict[self.label_key] = self.__unsqueeze__(
            data_dict[self.data_key], data_dict[self.label_key]
        )
        return data_dict


class RemoveBatchDimension(YuccaTransform):
    def __init__(self, data_key="image", label_key="label"):
        self.data_key = data_key
        self.label_key = label_key

    @staticmethod
    def get_params():
        pass

    def __squeeze__(self, data, label):
        data = data[0]
        if isinstance(label, list):
            label = [s[0] for s in label]
        else:
            label = label[0]
        return data, label

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        data_dict[self.data_key], data_dict[self.label_key] = self.__squeeze__(
            data_dict[self.data_key], data_dict[self.label_key]
        )
        return data_dict
