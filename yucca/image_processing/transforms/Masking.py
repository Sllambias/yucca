import numpy as np
import logging
from yucca.image_processing.transforms.YuccaTransform import YuccaTransform
from typing import Union, Iterable


class Masking(YuccaTransform):
    """
    A Yucca transform class for applying masking to input images or image volumes.

    It can be used for masking with a fixed ratio and fixed token size but it can also be used
    to mask a random ratio of each batch and with random token sizes.

    Args:
        mask (bool, optional): Whether to apply masking. Defaults to False.
        data_key (str, optional): Key to access the image data in the input dictionary. Defaults to "image".
        ratio (Union[Iterable[float], float], optional): Ratio of elements to be masked. Can be a single float or a range of floats.
            Defaults to 0.25. If it's a range of floats the masking ratio will be uniformly sampled from the two floats for each batch.
        patch_size (Union[Iterable[Union[int, float]], float, int], optional): Size of the mask token. Can be given either as exact
            patch size in the form of a single integer or an iterator of intergers, or it can be given as a ratio of the full volume.
            If patch_size is given as a single int (e.g. 4) the mask token size for a 2D image will be [4, 4] and for 3D [4,4,4].
            If patch_size is given as a single float (e.g. 0.05) the float will denote the ratio of the token size relative
                to the image volume size. For token size 0.05 with an image of shape [200, 100, 160] patch_size will be
                [200*0.05, 100*0.05, 160*0.05] = [10, 5, 8].
            If patch_size is given as an iterator of ints (e.g. [16,16,16]) it will remain unchanged.
            If patch_size is given as an iterator of two floats they will be uniformly sampled per batch, and then
                otherwise treated as the case with a single float.
    """

    def __init__(
        self,
        mask=False,
        data_key="image",
        ratio: Union[Iterable[float], float] = 0.25,
        patch_size: Union[Iterable[Union[float, int]], float, int] = 0.05,
        pixel_value: Union[float, int] = 0,
    ):
        self.mask = mask
        self.data_key = data_key
        self.ratio = ratio
        self.patch_size = patch_size
        self.pixel_value = pixel_value

    @staticmethod
    def get_params(input_shape, ratio, patch_size):
        if isinstance(ratio, (tuple, list)):
            # If ratio is a list/tuple it's a range of ratios from which me sample uniformly per batch
            ratio = np.random.uniform(*ratio)
        if isinstance(patch_size, float):
            # if patch_size is a float it will be treated as the ratio of the input size
            assert 1 > patch_size > 0, "if patch_size is a float it needs to be between 1 and 0."
            " It will be treated as a ratio to mask for each image dimension."
            if patch_size > 0.25:
                logging.warn(
                    "patch_size is set to a ratio over 0.25 of the image. "
                    "This is not intended and should be reconsidered."
                )
            patch_size = [int(i * patch_size) for i in input_shape]
        elif isinstance(patch_size, int):
            patch_size = [patch_size for i in input_shape]
        elif isinstance(patch_size, (tuple, list)) and isinstance(patch_size[0], float):
            # If we get a
            assert (
                len(patch_size) == 2 and np.all(1 > np.array(patch_size)) and np.all(np.array(patch_size) > 0)
            )
            if np.any(np.array(patch_size) > 0.25):
                logging.warn(
                    "patch_size is set to a ratio over 0.25 of the image. "
                    "This is not intended and should be reconsidered."
                )
            patch_size = [int(i * np.random.uniform(*patch_size)) for i in input_shape]
        return ratio, patch_size

    def __mask__(self, image_volume, ratio, patch_size):
        assert len(image_volume.shape[2:]) == len(patch_size), (
            "mask token size not compatible with input data"
            f"mask token is: {patch_size} and image is shape: {image_volume.shape[2:]}"
        )
        # np.ceil to get a grid with exact or larger dims than the input image
        # later we will crop it to the desired dims
        slices = [slice(0, shape) for shape in image_volume.shape[2:]]
        grid_dims = np.ceil(image_volume.shape[2:] / np.array(patch_size)).astype(np.uint8)

        grid_flat = np.ones(np.prod(grid_dims))
        grid_flat[: int(len(grid_flat) * ratio)] = 0
        np.random.shuffle(grid_flat)
        grid = grid_flat.reshape(grid_dims)
        for idx, size in enumerate(patch_size):
            grid = np.repeat(grid, repeats=size, axis=idx)

        image_volume[:, :, grid[*slices] == 0] = self.pixel_value
        return image_volume

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"
        if self.mask:
            ratio, patch_size = self.get_params(
                data_dict[self.data_key].shape[2:], self.ratio, self.patch_size
            )
            data_dict[self.data_key] = self.__mask__(data_dict[self.data_key], ratio, patch_size)
        return data_dict

