"""
https://www.meccanismocomplesso.org/en/3d-rotations-and-euler-angles-in-python/
From: https://stackoverflow.com/questions/59738230/apply-rotation-defined-by-euler-angles-to-3d-image-in-python
"""

import numpy as np
from yucca.modules.data.augmentation.transforms.YuccaTransform import YuccaTransform
from typing import Tuple
from yucca.functional.transforms import spatial, torch_spatial


class Spatial(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        label_key="label",
        crop=False,
        cval="min",
        order=3,
        patch_size: Tuple[int] = None,
        clip_to_input_range=True,
        random_crop=True,
        p_deform_per_sample: float = 1.0,
        deform_sigma=(20, 30),
        deform_alpha=(300, 600),
        p_rot_per_sample: float = 1.0,
        p_rot_per_axis: float = 1.0,
        x_rot_in_degrees=(0.0, 10.0),
        y_rot_in_degrees=(0.0, 10.0),
        z_rot_in_degrees=(0.0, 10.0),
        p_scale_per_sample: float = 1.0,
        scale_factor=(0.85, 1.15),
        skip_label=False,
    ):
        self.data_key = data_key
        self.label_key = label_key
        self.skip_label = skip_label
        self.order = order
        self.do_crop = crop
        self.cval = cval
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.clip_to_input_range = clip_to_input_range

        self.p_deform_per_sample = p_deform_per_sample
        self.deform_sigma = deform_sigma
        self.deform_alpha = deform_alpha

        self.p_rot_per_sample = p_rot_per_sample
        self.p_rot_per_axis = p_rot_per_axis
        self.x_rot_in_degrees = x_rot_in_degrees
        self.y_rot_in_degrees = y_rot_in_degrees
        self.z_rot_in_degrees = z_rot_in_degrees

        self.p_scale_per_sample = p_scale_per_sample
        self.scale_factor = scale_factor

    @staticmethod
    def get_params(
        deform_alpha: Tuple[float],
        deform_sigma: Tuple[float],
        x_rot: Tuple[float],
        y_rot: Tuple[float],
        z_rot: Tuple[float],
        scale_factor: Tuple[float],
    ) -> Tuple[float]:
        if deform_alpha:
            deform_alpha = float(np.random.uniform(*deform_alpha))
        if deform_sigma:
            deform_sigma = float(np.random.uniform(*deform_sigma))

        if x_rot:
            x_rot = float(np.random.uniform(*x_rot)) * (np.pi / 180)
        if y_rot:
            y_rot = float(np.random.uniform(*y_rot)) * (np.pi / 180)
        if z_rot:
            z_rot = float(np.random.uniform(*z_rot)) * (np.pi / 180)

        if scale_factor:
            scale_factor = float(np.random.uniform(*scale_factor))

        return deform_alpha, deform_sigma, x_rot, y_rot, z_rot, scale_factor

    def __CropDeformRotateScale__(
        self,
        data_dict,
        patch_size,
        alpha,
        sigma,
        x_rot,
        y_rot,
        z_rot,
        scale_factor,
    ):
        image, label = spatial(
            image=data_dict[self.data_key],
            patch_size=patch_size,
            p_deform=self.p_deform_per_sample,
            p_rot=self.p_rot_per_sample,
            p_rot_per_axis=self.p_rot_per_axis,
            p_scale=self.p_scale_per_sample,
            alpha=alpha,
            sigma=sigma,
            x_rot=x_rot,
            y_rot=y_rot,
            z_rot=z_rot,
            scale_factor=scale_factor,
            clip_to_input_range=self.clip_to_input_range,
            label=data_dict.get(self.label_key),
            skip_label=self.skip_label,
            do_crop=self.do_crop,
            random_crop=self.random_crop,
            order=self.order,
            cval=self.cval,
        )
        data_dict[self.data_key] = image
        if label is not None and not self.skip_label:
            data_dict[self.label_key] = label
        return data_dict

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
			\nShould be (c, x, y, z) or (c, x, y) and is: {data_dict[self.data_key].shape}"

        (
            deform_alpha,
            deform_sigma,
            x_rot_rad,
            y_rot_rad,
            z_rot_rad,
            scale_factor,
        ) = self.get_params(
            deform_alpha=self.deform_alpha,
            deform_sigma=self.deform_sigma,
            x_rot=self.x_rot_in_degrees,
            y_rot=self.y_rot_in_degrees,
            z_rot=self.z_rot_in_degrees,
            scale_factor=self.scale_factor,
        )

        data_dict = self.__CropDeformRotateScale__(
            data_dict,
            self.patch_size,
            deform_alpha,
            deform_sigma,
            x_rot_rad,
            y_rot_rad,
            z_rot_rad,
            scale_factor,
        )
        return data_dict


class Torch_Spatial(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        label_key="label",
        crop=True,
        interpolation_mode="bilinear",
        patch_size: Tuple[int] = None,
        clip_to_input_range=True,
        random_crop=True,
        p_deform_all_channel: float = 1.0,
        deform_sigma=(5, 20),
        deform_alpha=(5, 20),
        p_rot_all_channel: float = 1.0,
        p_rot_per_axis: float = 1.0,
        x_rot_in_degrees=(0.0, 10.0),
        y_rot_in_degrees=(0.0, 10.0),
        z_rot_in_degrees=(0.0, 10.0),
        p_scale_all_channel: float = 1.0,
        scale_factor=(0.85, 1.15),
        skip_label=False,
    ):
        self.data_key = data_key
        self.label_key = label_key
        self.skip_label = skip_label
        self.interpolation_mode = interpolation_mode
        self.do_crop = crop
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.clip_to_input_range = clip_to_input_range

        self.p_deform_all_channel = p_deform_all_channel
        self.deform_sigma = deform_sigma
        self.deform_alpha = deform_alpha

        self.p_rot_all_channel = p_rot_all_channel
        self.p_rot_per_axis = p_rot_per_axis
        self.x_rot_in_degrees = x_rot_in_degrees
        self.y_rot_in_degrees = y_rot_in_degrees
        self.z_rot_in_degrees = z_rot_in_degrees

        self.p_scale_all_channel = p_scale_all_channel
        self.scale_factor = scale_factor

    @staticmethod
    def get_params(
        deform_alpha: Tuple[float],
        deform_sigma: Tuple[float],
        x_rot: Tuple[float],
        y_rot: Tuple[float],
        z_rot: Tuple[float],
        scale_factor: Tuple[float],
    ) -> Tuple[float]:
        if deform_alpha:
            deform_alpha = float(np.random.uniform(*deform_alpha))
        if deform_sigma:
            deform_sigma = float(np.random.uniform(*deform_sigma))

        if x_rot:
            x_rot = float(np.random.uniform(*x_rot)) * (np.pi / 180)
        if y_rot:
            y_rot = float(np.random.uniform(*y_rot)) * (np.pi / 180)
        if z_rot:
            z_rot = float(np.random.uniform(*z_rot)) * (np.pi / 180)

        if scale_factor:
            scale_factor = float(np.random.uniform(*scale_factor))

        return deform_alpha, deform_sigma, x_rot, y_rot, z_rot, scale_factor

    def __CropDeformRotateScale__(
        self,
        data_dict,
        patch_size,
        alpha,
        sigma,
        x_rot,
        y_rot,
        z_rot,
        scale_factor,
    ):
        image, label = torch_spatial(
            image=data_dict[self.data_key],
            patch_size=patch_size,
            p_deform=self.p_deform_all_channel,
            p_rot=self.p_rot_all_channel,
            p_rot_per_axis=self.p_rot_per_axis,
            p_scale=self.p_scale_all_channel,
            alpha=alpha,
            sigma=sigma,
            x_rot=x_rot,
            y_rot=y_rot,
            z_rot=z_rot,
            scale_factor=scale_factor,
            clip_to_input_range=self.clip_to_input_range,
            label=data_dict.get(self.label_key),
            skip_label=self.skip_label,
            do_crop=self.do_crop,
            random_crop=self.random_crop,
            interpolation_mode=self.interpolation_mode,
        )
        data_dict[self.data_key] = image
        if label is not None and not self.skip_label:
            data_dict[self.label_key] = label
        return data_dict

    def __call__(self, data_dict):
        (
            deform_alpha,
            deform_sigma,
            x_rot_rad,
            y_rot_rad,
            z_rot_rad,
            scale_factor,
        ) = self.get_params(
            deform_alpha=self.deform_alpha,
            deform_sigma=self.deform_sigma,
            x_rot=self.x_rot_in_degrees,
            y_rot=self.y_rot_in_degrees,
            z_rot=self.z_rot_in_degrees,
            scale_factor=self.scale_factor,
        )

        data_dict = self.__CropDeformRotateScale__(
            data_dict,
            self.patch_size,
            deform_alpha,
            deform_sigma,
            x_rot_rad,
            y_rot_rad,
            z_rot_rad,
            scale_factor,
        )
        return data_dict
