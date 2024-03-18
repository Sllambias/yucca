"""
https://www.meccanismocomplesso.org/en/3d-rotations-and-euler-angles-in-python/
From: https://stackoverflow.com/questions/59738230/apply-rotation-defined-by-euler-angles-to-3d-image-in-python
"""

import numpy as np
from scipy.ndimage import map_coordinates
from yucca.image_processing.matrix_ops import (
    create_zero_centered_coordinate_matrix,
    deform_coordinate_matrix,
    Rx,
    Ry,
    Rz,
    Rz2D,
)
from yucca.image_processing.transforms.YuccaTransform import YuccaTransform
from typing import Tuple


class Spatial(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        label_key="label",
        crop=False,
        cval="min",
        patch_size: Tuple[int] = None,
        random_crop=True,
        p_deform_per_sample=1,
        deform_sigma=(20, 30),
        deform_alpha=(300, 600),
        p_rot_per_sample=1,
        p_rot_per_axis=1,
        x_rot_in_degrees=(0.0, 10.0),
        y_rot_in_degrees=(0.0, 10.0),
        z_rot_in_degrees=(0.0, 10.0),
        p_scale_per_sample=1,
        scale_factor=(0.85, 1.15),
        skip_label=False,
    ):
        self.data_key = data_key
        self.label_key = label_key
        self.skip_label = skip_label
        self.do_crop = crop
        self.cval = cval
        self.patch_size = patch_size
        self.random_crop = random_crop

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
        skip_label,
    ):
        image = data_dict[self.data_key]
        if not self.do_crop:
            patch_size = image.shape[2:]
        if self.cval == "min":
            cval = float(image.min())
        else:
            cval = self.cval
        assert isinstance(cval, (int, float)), f"got {cval} of type {type(cval)}"

        coords = create_zero_centered_coordinate_matrix(patch_size)
        imageCanvas = np.zeros((image.shape[0], image.shape[1], *patch_size), dtype=np.float32)

        # First we apply deformation to the coordinate matrix
        if np.random.uniform() < self.p_deform_per_sample:
            coords = deform_coordinate_matrix(coords, alpha=alpha, sigma=sigma)

        # Then we rotate the coordinate matrix around one or more axes
        if np.random.uniform() < self.p_rot_per_sample:
            rot_matrix = np.eye(len(patch_size))
            if len(patch_size) == 2:
                rot_matrix = np.dot(rot_matrix, Rz2D(z_rot))
            else:
                if np.random.uniform() < self.p_rot_per_axis:
                    rot_matrix = np.dot(rot_matrix, Rx(x_rot))
                if np.random.uniform() < self.p_rot_per_axis:
                    rot_matrix = np.dot(rot_matrix, Ry(y_rot))
                if np.random.uniform() < self.p_rot_per_axis:
                    rot_matrix = np.dot(rot_matrix, Rz(z_rot))

            coords = np.dot(coords.reshape(len(patch_size), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)

        # And finally scale it
        # Scaling effect is "inverted"
        # i.e. a scale factor of 0.9 will zoom in
        if np.random.uniform() < self.p_scale_per_sample:
            coords *= scale_factor

        if self.random_crop and self.do_crop:
            for d in range(len(patch_size)):
                crop_center_idx = [
                    np.random.randint(
                        int(patch_size[d] / 2),
                        image.shape[d + 2] - int(patch_size[d] / 2) + 1,
                    )
                ]
                coords[d] += crop_center_idx
        else:
            # Reversing the zero-centering of the coordinates
            for d in range(len(patch_size)):
                coords[d] += image.shape[d + 2] / 2.0 - 0.5

        # Mapping the images to the distorted coordinates
        for b in range(image.shape[0]):
            for c in range(image.shape[1]):
                imageCanvas[b, c] = map_coordinates(
                    image[b, c].astype(float),
                    coords,
                    order=3,
                    mode="constant",
                    cval=cval,
                ).astype(image.dtype)

        data_dict[self.data_key] = imageCanvas
        if data_dict.get(self.label_key) is not None and not skip_label:
            label = data_dict.get(self.label_key)
            labelCanvas = np.zeros(
                (label.shape[0], label.shape[1], *patch_size),
                dtype=np.float32,
            )

            # Mapping the labelmentations to the distorted coordinates
            for b in range(label.shape[0]):
                for c in range(label.shape[1]):
                    labelCanvas[b, c] = map_coordinates(label[b, c], coords, order=0, mode="constant", cval=0.0).astype(
                        label.dtype
                    )
            data_dict[self.label_key] = labelCanvas
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
            self.skip_label,
        )
        return data_dict
