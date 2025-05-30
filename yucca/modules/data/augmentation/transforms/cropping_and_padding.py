import numpy as np
import torch
from yucca.modules.data.augmentation.transforms.YuccaTransform import YuccaTransform
from yucca.functional.transforms.croppad import croppad
from yucca.functional.transforms.torch import torch_croppad
from typing import Literal, Union


class CropPad(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        label_key="label",
        pad_value: Union[Literal["min", "zero", "edge"], int, float] = "min",
        patch_size: tuple | list = None,
        p_oversample_foreground=0.0,
    ):
        self.data_key = data_key
        self.label_key = label_key
        self.pad_value = pad_value
        self.patch_size = patch_size
        self.p_oversample_foreground = p_oversample_foreground

    @staticmethod
    def get_params(data, pad_value, target_shape):
        if pad_value == "min":
            pad_kwargs = {"constant_values": data.min(), "mode": "constant"}
        elif pad_value == "zero":
            pad_kwargs = {"constant_values": np.zeros(1, dtype=data.dtype), "mode": "constant"}
        elif isinstance(pad_value, int) or isinstance(pad_value, float):
            pad_kwargs = {"constant_values": pad_value, "mode": "constant"}
        elif pad_value == "edge":
            pad_kwargs = {"mode": "edge"}
        else:
            print("Unrecognized pad value detected.")
        input_shape = data.shape
        target_image_shape = (input_shape[0], *target_shape)
        target_label_shape = (1, *target_shape)
        return input_shape, target_image_shape, target_label_shape, pad_kwargs

    def __croppad__(
        self,
        data_dict: np.ndarray,
        image_properties: dict,
        input_shape: np.ndarray,
        p_oversample_foreground: float,
        target_image_shape: list | tuple,
        target_label_shape: list | tuple,
        **pad_kwargs,
    ):
        image = data_dict[self.data_key]
        if data_dict.get(self.label_key) is not None:
            label = data_dict[self.label_key]
        else:
            label = None

        image, label = croppad(
            image=image,
            image_properties=image_properties,
            label=label,
            input_dims=len(input_shape[1:]),
            patch_size=self.patch_size,
            p_oversample_foreground=p_oversample_foreground,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
            **pad_kwargs,
        )

        data_dict[self.data_key] = image
        if label is not None:
            data_dict[self.label_key] = label
        return data_dict

    def __call__(self, packed_data_dict=None, image_properties=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict

        input_shape, target_image_shape, target_label_shape, pad_kwargs = self.get_params(
            data=data_dict[self.data_key], pad_value=self.pad_value, target_shape=self.patch_size
        )

        data_dict = self.__croppad__(
            data_dict=data_dict,
            image_properties=image_properties,
            input_shape=input_shape,
            p_oversample_foreground=self.p_oversample_foreground,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
            **pad_kwargs,
        )
        return data_dict


class Torch_CropPad(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        label_key="label",
        pad_value: Union[Literal["min", "zero", "replicate"], int, float] = "min",
        patch_size: tuple | list = None,
        p_oversample_foreground=0.0,
    ):
        self.data_key = data_key
        self.label_key = label_key
        self.pad_value = pad_value
        self.patch_size = patch_size
        self.p_oversample_foreground = p_oversample_foreground

    @staticmethod
    def get_params(data, pad_value, target_shape):
        if pad_value == "min":
            pad_kwargs = {"value": data.min(), "mode": "constant"}
        elif pad_value == "zero":
            pad_kwargs = {"value": torch.zeros(1, dtype=data.dtype), "mode": "constant"}
        elif isinstance(pad_value, int) or isinstance(pad_value, float):
            pad_kwargs = {"value": pad_value, "mode": "constant"}
        elif pad_value == "replicate":
            pad_kwargs = {"mode": "replicate"}
        else:
            print("Unrecognized pad value detected.")
        input_shape = data.shape
        target_image_shape = (input_shape[0], *target_shape)
        target_label_shape = (1, *target_shape)
        return input_shape, target_image_shape, target_label_shape, pad_kwargs

    def __croppad__(
        self,
        data_dict: np.ndarray,
        foreground_locations: dict,
        input_shape: np.ndarray,
        p_oversample_foreground: float,
        target_image_shape: list | tuple,
        target_label_shape: list | tuple,
        **pad_kwargs,
    ):
        image = data_dict[self.data_key]
        if data_dict.get(self.label_key) is not None:
            label = data_dict[self.label_key]
        else:
            label = None

        image, label = torch_croppad(
            image=image,
            input_dims=len(input_shape[1:]),
            patch_size=self.patch_size,
            foreground_locations=foreground_locations,
            label=label,
            p_oversample_foreground=p_oversample_foreground,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
            **pad_kwargs,
        )

        data_dict[self.data_key] = image
        if label is not None:
            data_dict[self.label_key] = label
        return data_dict

    def __call__(self, data_dict, foreground_locations=[]):
        input_shape, target_image_shape, target_label_shape, pad_kwargs = self.get_params(
            data=data_dict[self.data_key], pad_value=self.pad_value, target_shape=self.patch_size
        )

        data_dict = self.__croppad__(
            data_dict=data_dict,
            foreground_locations=foreground_locations,
            input_shape=input_shape,
            p_oversample_foreground=self.p_oversample_foreground,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
            **pad_kwargs,
        )
        return data_dict
