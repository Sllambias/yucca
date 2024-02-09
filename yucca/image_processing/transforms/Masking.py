import numpy as np
import logging
from yucca.image_processing.transforms.YuccaTransform import YuccaTransform
from typing import Union, Tuple, List, Optional, Iterable

class Masking(YuccaTransform):
    """
    A Yucca transform class for applying masking to input images or image volumes.

    It can be used for masking with a fixed ratio and fixed token size but it can also be used 
    to mask a random ratio of each batch and with random token sizes.

    Args:
        mask (bool, optional): Whether to apply masking. Defaults to False.
        data_key (str, optional): Key to access the image data in the input dictionary. Defaults to "image".
        mask_ratio (Union[Iterable[float], float], optional): Ratio of elements to be masked. Can be a single float or a range of floats.
            Defaults to 0.25. If it's a range of floats the masking ratio will be uniformly sampled from the two floats for each batch.
        mask_token_size (Union[Iterable[Union[int, float]], float, int], optional): Size of the mask token. Can be a single integer,
            a single float representing a ratio, or a range of integers/floats. Defaults to 0.05.
            If mask_token_size is given as a single int (e.g. 4) the mask token size for a 2D image will be [4, 4] and for 3D [4,4,4].
            If mask_token_size is given as a single float (e.g. 0.05) the float will denote the ratio of the token size relative
                to the image volume size. For token size 0.05 with an image of shape [200, 100, 160] mask_token_size will be
                [200*0.05, 100*0.05, 160*0.05] = [10, 5, 8].
            If mask_token_size is given as an iterator of ints (e.g. [16,16,16]) it will remain unchanged.
            if mask_token_size is given as an iterator of two floats they will be uniformly sampled per batch, and then
                otherwise treated as the above case with a single float.

    """

    def __init__(self, mask=False, data_key="image", mask_ratio: Union[Iterable[float], float] = 0.25, mask_token_size: Union[Iterable[Union[int, float]], float, int] = 0.05):
        self.mask = mask
        self.data_key = data_key
        self.mask_ratio = mask_ratio
        self.mask_token_size = mask_token_size

    @staticmethod
    def get_params(input_shape, mask_ratio, mask_token_size):
        if isinstance(mask_ratio, (tuple, list)):
            # If ratio is a list/tuple it's a range of ratios from which me sample uniformly per batch
            mask_ratio = np.random.uniform(*mask_ratio)
        if isinstance(mask_token_size, float):
            # if mask_token_size is a float it will be treated as the ratio of the input size
            assert 1 > mask_token_size > 0, "if mask_token_size is a float it needs to be between 1 and 0."
            " It will be treated as a ratio to mask for each image dimension."
            if mask_token_size > 0.25:
                logging.warn("mask_token_size is set to a ratio over 0.25 of the image. "
                             "This is not intended and should be reconsidered.")
            mask_token_size = [int(i * mask_token_size) for i in input_shape]
            print(mask_token_size)
        elif isinstance(mask_token_size, int):
            mask_token_size = [mask_token_size for i in input_shape]
        elif isinstance(mask_token_size, (tuple, list)) and isinstance(mask_token_size[0], float):
            # If we get a 
            assert len(mask_token_size) == 2 and np.all(1 > np.array(mask_token_size)) and np.all(np.array(mask_token_size) > 0)
            if np.any(np.array(mask_token_size) > 0.25):
                logging.warn("mask_token_size is set to a ratio over 0.25 of the image. "
                             "This is not intended and should be reconsidered.")
            mask_token_size = [int(i * np.random.uniform(*mask_token_size)) for i in input_shape]
        return mask_ratio, mask_token_size
    
    def __mask__(self, image_volume, mask_ratio, mask_token_size):
        assert len(image_volume.shape[2:]) == len(mask_token_size), (
            "mask token size not compatible with input data"
            f"mask token is: {mask_token_size} and image is shape: {image_volume.shape[2:]}"
        )
        # np.ceil to get a grid with exact or larger dims than the input image
        # later we will crop it to the desired dims
        slices = [slice(0, shape) for shape in image_volume.shape[2:]]
        grid_dims = np.ceil(image_volume.shape[2:] / np.array(mask_token_size)).astype(np.uint8)

        grid_flat = np.ones(np.prod(grid_dims))
        grid_flat[: int(len(grid_flat) * mask_ratio)] = 0
        np.random.shuffle(grid_flat)
        grid = grid_flat.reshape(grid_dims)
        for idx, size in enumerate(mask_token_size):
            grid = np.repeat(grid, repeats=size, axis=idx)
        
        image_volume = np.multiply(image_volume, grid[*slices])
        return image_volume

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"
        if self.mask:
            mask_ratio, mask_token_size= self.get_params(data_dict[self.data_key].shape[2:], self.mask_ratio, self.mask_token_size)
            data_dict[self.data_key] = self.__mask__(data_dict[self.data_key], mask_ratio, mask_token_size)
        return data_dict

