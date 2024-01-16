import numpy as np


def crop_to_box(array, bbox):
    """
    Crops a 3D array to the Bounding Box indices
    Should be a list of [xmin, xmax, ymin, ymax (, zmin, zmax)]
    """
    if len(bbox) > 5:
        bbox_slices = (
            slice(bbox[0], bbox[1]),
            slice(bbox[2], bbox[3]),
            slice(bbox[4], bbox[5]),
        )
    else:
        bbox_slices = (slice(bbox[0], bbox[1]), slice(bbox[2], bbox[3]))
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
