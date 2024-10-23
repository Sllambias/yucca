import numpy as np
import torch
import torch.nn.functional as F


def crop_to_box(array, bbox):
    """
    Crops an array to the Bounding Box indices
    Should be a list of [xmin, xmax, ymin, ymax (, zmin, zmax)]

    We add +1 because slicing excludes the high val index. Which it should not do here.
    """
    if len(bbox) > 5:
        bbox_slices = (
            slice(bbox[0], bbox[1] + 1),
            slice(bbox[2], bbox[3] + 1),
            slice(bbox[4], bbox[5] + 1),
        )
    else:
        bbox_slices = (slice(bbox[0], bbox[1] + 1), slice(bbox[2], bbox[3] + 1))
    return array[bbox_slices]


def pad_to_size(array, size, **kwargs):
    pad_box = get_pad_box(array, size)
    if len(pad_box) > 5:
        array_padded = np.pad(
            array,
            (
                (pad_box[0], pad_box[1]),
                (pad_box[2], pad_box[3]),
                (pad_box[4], pad_box[5]),
            ),
            **kwargs,
        )
        return array_padded, pad_box

    array_padded = np.pad(
        array,
        ((pad_box[0], pad_box[1]), (pad_box[2], pad_box[3])),
        **kwargs,
    )
    return array_padded, pad_box


def get_pad_box(original_array, min_size):
    assert len(original_array.shape) in [2, 3], "incorrect array shape"
    assert len(min_size) in [2, 3], "incorrect patch_size shape"
    pad_box = []

    if len(original_array.shape) == 3:
        if len(min_size) == 2:
            # 3D Data for 2D model.
            min_size = (0, *min_size)
        for i in range(3):
            val = max(0, min_size[i] - original_array.shape[i])
            pad_box.append(val // 2)
            pad_box.append(val // 2 + val % 2)
        return pad_box

    # 2D data + 2D model
    for i in range(2):
        val = max(0, min_size[i] - original_array.shape[i])
        pad_box.append(val // 2)
        pad_box.append(val // 2 + val % 2)
    return pad_box


def get_pad_kwargs(data, pad_value):
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
    return pad_kwargs


def ensure_batch_fits_patch_size(batch, patch_size):
    """
    Pads the spatial dimensions of the input tensor so that they are at least the size of the patch dimensions.
    If all spatial dimensions are already larger than or equal to the patch size, the input tensor is returned unchanged.

    Parameters:
    - batch: dict
        a dict with keys {"data": data, "data_properties": data_properties, "case_id": case_id},
        where data is a Tensor of shape (B, C, *spatial_dims)

    - patch_size: tuple of ints
        The minimum desired size for each spatial dimension.

    Returns:
    - padded_input: torch.Tensor
        The input tensor padded to the desired spatial dimensions.
    """
    image = batch["data"]
    image_properties = batch["data_properties"]
    spatial_dims = image.dim() - 2  # Subtract batch and channel dimensions

    if spatial_dims == len(patch_size) + 1:  # 2D patches from 3D data
        patch_size = (1,) + patch_size

    if spatial_dims != len(patch_size):
        raise ValueError(
            f"Input spatial dimensions and patch size dimensions do not match. Got patch_size: {patch_size} and spatial_dims: {spatial_dims} from image of shape: {image.shape}"
        )

    current_sizes = image.shape[2:]  # Spatial dimensions

    current_sizes_tensor = torch.tensor(current_sizes)
    patch_size_tensor = torch.tensor(patch_size)

    pad_sizes = torch.clamp(patch_size_tensor - current_sizes_tensor, min=0)
    pad_left = pad_sizes // 2
    pad_right = pad_sizes - pad_left

    # Construct padding tuple in reverse order for F.pad
    padding_reversed = []
    for left, right in zip(reversed(pad_left.tolist()), reversed(pad_right.tolist())):
        padding_reversed.extend([left, right])

    padded_input = F.pad(image, padding_reversed)

    image_properties["padded_shape"] = np.array(padded_input.shape[2:])
    image_properties["padding"] = list(reversed(padding_reversed))

    return padded_input, image_properties
