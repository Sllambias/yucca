import numpy as np
from typing import Optional, Union
from yucca.functional.array_operations.matrix_ops import (
    create_zero_centered_coordinate_matrix,
    deform_coordinate_matrix,
    Rx,
    Ry,
    Rz,
    Rz2D,
)
from scipy.ndimage import map_coordinates


def spatial(
    image,
    patch_size,
    p_deform,
    p_rot,
    p_rot_per_axis,
    p_scale,
    alpha,
    sigma,
    x_rot,
    y_rot,
    z_rot,
    scale_factor,
    clip_to_input_range,
    label: Optional[np.ndarray] = None,
    skip_label: bool = False,
    do_crop: bool = True,
    random_crop: bool = True,
    order: int = 3,
    cval: Optional[Union[str, int, float]] = "min",
):
    if not do_crop:
        patch_size = image.shape[2:]
    if cval == "min":
        cval = float(image.min())
    else:
        cval = cval
    assert isinstance(cval, (int, float)), f"got {cval} of type {type(cval)}"

    coords = create_zero_centered_coordinate_matrix(patch_size)
    image_canvas = np.zeros((image.shape[0], image.shape[1], *patch_size), dtype=np.float32)

    # First we apply deformation to the coordinate matrix
    if np.random.uniform() < p_deform:
        coords = deform_coordinate_matrix(coords, alpha=alpha, sigma=sigma)

    # Then we rotate the coordinate matrix around one or more axes
    if np.random.uniform() < p_rot:
        rot_matrix = np.eye(len(patch_size))
        if len(patch_size) == 2:
            rot_matrix = np.dot(rot_matrix, Rz2D(z_rot))
        else:
            if np.random.uniform() < p_rot_per_axis:
                rot_matrix = np.dot(rot_matrix, Rx(x_rot))
            if np.random.uniform() < p_rot_per_axis:
                rot_matrix = np.dot(rot_matrix, Ry(y_rot))
            if np.random.uniform() < p_rot_per_axis:
                rot_matrix = np.dot(rot_matrix, Rz(z_rot))

        coords = np.dot(coords.reshape(len(patch_size), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)

    # And finally scale it
    # Scaling effect is "inverted"
    # i.e. a scale factor of 0.9 will zoom in
    if np.random.uniform() < p_scale:
        coords *= scale_factor

    if random_crop and do_crop:
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
            img_min = image.min()
            img_max = image.max()

            image_canvas[b, c] = map_coordinates(
                image[b, c].astype(float),
                coords,
                order=order,
                mode="constant",
                cval=cval,
            ).astype(image.dtype)

            if clip_to_input_range:
                image_canvas[b, c] = np.clip(image_canvas[b, c], a_min=img_min, a_max=img_max)

    if label is not None and not skip_label:
        label_canvas = np.zeros(
            (label.shape[0], label.shape[1], *patch_size),
            dtype=np.float32,
        )

        # Mapping the labelmentations to the distorted coordinates
        for b in range(label.shape[0]):
            for c in range(label.shape[1]):
                label_canvas[b, c] = map_coordinates(label[b, c], coords, order=0, mode="constant", cval=0.0).astype(
                    label.dtype
                )
        return image_canvas, label_canvas
    return image_canvas, None
