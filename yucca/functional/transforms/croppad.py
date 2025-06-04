import numpy as np


def croppad(
    image: np.ndarray,
    image_properties: dict,
    input_dims: np.ndarray,
    patch_size,
    p_oversample_foreground,
    target_image_shape: list | tuple,
    target_label_shape: list | tuple,
    label: np.ndarray = None,
    **pad_kwargs,
):
    if len(patch_size) == 3:
        image, label = croppad_3D_case_from_3D(
            image=image,
            image_properties=image_properties,
            label=label,
            patch_size=patch_size,
            p_oversample_foreground=p_oversample_foreground,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
            **pad_kwargs,
        )
    elif len(patch_size) == 2 and input_dims == 3:
        image, label = croppad_2D_case_from_3D(
            image=image,
            image_properties=image_properties,
            label=label,
            patch_size=patch_size,
            p_oversample_foreground=p_oversample_foreground,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
            **pad_kwargs,
        )
    elif len(patch_size) == 2 and input_dims == 2:
        image, label = croppad_2D_case_from_2D(
            image=image,
            image_properties=image_properties,
            label=label,
            patch_size=patch_size,
            p_oversample_foreground=p_oversample_foreground,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
            **pad_kwargs,
        )

    return image, label


def croppad_3D_case_from_3D(
    image, image_properties, label, patch_size, p_oversample_foreground, target_image_shape, target_label_shape, **pad_kwargs
):
    image_out = np.zeros(target_image_shape)
    label_out = np.zeros(target_label_shape)

    # First we pad to ensure min size is met
    to_pad = []
    for d in range(3):
        if image.shape[d + 1] < patch_size[d]:
            to_pad += [patch_size[d] - image.shape[d + 1]]
        else:
            to_pad += [0]

    pad_lb_x = to_pad[0] // 2
    pad_ub_x = to_pad[0] // 2 + to_pad[0] % 2
    pad_lb_y = to_pad[1] // 2
    pad_ub_y = to_pad[1] // 2 + to_pad[1] % 2
    pad_lb_z = to_pad[2] // 2
    pad_ub_z = to_pad[2] // 2 + to_pad[2] % 2

    # This is where we should implement any patch selection biases.
    # The final patch excted after augmentation will always be the center of this patch
    # to avoid interpolation artifacts near the borders
    crop_start_idx = []
    if len(image_properties["foreground_locations"]) == 0 or np.random.uniform() >= p_oversample_foreground:
        for d in range(3):
            if image.shape[d + 1] < patch_size[d]:
                crop_start_idx += [0]
            else:
                crop_start_idx += [np.random.randint(image.shape[d + 1] - patch_size[d] + 1)]
    else:
        location = select_foreground_voxel_to_include(image_properties)
        for d in range(3):
            if image.shape[d + 1] < patch_size[d]:
                crop_start_idx += [0]
            else:
                crop_start_idx += [
                    np.random.randint(
                        max(0, location[d] - patch_size[d]),
                        min(location[d], image.shape[d + 1] - patch_size[d]) + 1,
                    )
                ]

    image_out[
        :,
        :,
        :,
        :,
    ] = np.pad(
        image[
            :,
            crop_start_idx[0] : crop_start_idx[0] + patch_size[0],
            crop_start_idx[1] : crop_start_idx[1] + patch_size[1],
            crop_start_idx[2] : crop_start_idx[2] + patch_size[2],
        ],
        ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)),
        **pad_kwargs,
    )
    if label is None:
        return image_out, None
    label_out[
        :,
        :,
        :,
        :,
    ] = np.pad(
        label[
            :,
            crop_start_idx[0] : crop_start_idx[0] + patch_size[0],
            crop_start_idx[1] : crop_start_idx[1] + patch_size[1],
            crop_start_idx[2] : crop_start_idx[2] + patch_size[2],
        ],
        ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)),
    )
    return image_out, label_out


def croppad_2D_case_from_3D(
    image, image_properties, label, patch_size, p_oversample_foreground, target_image_shape, target_label_shape, **pad_kwargs
):
    """
    The possible input for this can be 2D or 3D data.
    For 2D we want to pad or crop as necessary.
    For 3D we want to first select a slice from the first dimension, i.e. volume[idx, :, :],
    then pad or crop as necessary.
    """
    image_out = np.zeros(target_image_shape)
    label_out = np.zeros(target_label_shape)

    # First we pad to ensure min size is met
    to_pad = []
    for d in range(2):
        if image.shape[d + 2] < patch_size[d]:
            to_pad += [patch_size[d] - image.shape[d + 2]]
        else:
            to_pad += [0]

    pad_lb_y = to_pad[0] // 2
    pad_ub_y = to_pad[0] // 2 + to_pad[0] % 2
    pad_lb_z = to_pad[1] // 2
    pad_ub_z = to_pad[1] // 2 + to_pad[1] % 2

    # This is where we should implement any patch selection biases.
    # The final patch extracted after augmentation will always be the center of this patch
    # as this is where augmentation-induced interpolation artefacts are least likely
    crop_start_idx = []
    if len(image_properties["foreground_locations"]) == 0 or np.random.uniform() >= p_oversample_foreground:
        x_idx = np.random.randint(image.shape[1])
        for d in range(2):
            if image.shape[d + 2] < patch_size[d]:
                crop_start_idx += [0]
            else:
                crop_start_idx += [np.random.randint(image.shape[d + 2] - patch_size[d] + 1)]
    else:
        location = select_foreground_voxel_to_include(image_properties)
        x_idx = location[0]
        for d in range(2):
            if image.shape[d + 2] < patch_size[d]:
                crop_start_idx += [0]
            else:
                crop_start_idx += [
                    np.random.randint(
                        max(0, location[d + 1] - patch_size[d]),
                        min(location[d + 1], image.shape[d + 2] - patch_size[d]) + 1,
                    )
                ]

    image_out[:, :, :] = np.pad(
        image[
            :,
            x_idx,
            crop_start_idx[0] : crop_start_idx[0] + patch_size[0],
            crop_start_idx[1] : crop_start_idx[1] + patch_size[1],
        ],
        ((0, 0), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)),
        **pad_kwargs,
    )

    if label is None:
        return image_out, None

    label_out[:, :, :] = np.pad(
        label[
            :,
            x_idx,
            crop_start_idx[0] : crop_start_idx[0] + patch_size[0],
            crop_start_idx[1] : crop_start_idx[1] + patch_size[1],
        ],
        ((0, 0), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)),
    )

    return image_out, label_out


def croppad_2D_case_from_2D(
    image, image_properties, label, patch_size, p_oversample_foreground, target_image_shape, target_label_shape, **pad_kwargs
):
    """
    The possible input for this can be 2D or 3D data.
    For 2D we want to pad or crop as necessary.
    For 3D we want to first select a slice from the first dimension, i.e. volume[idx, :, :],
    then pad or crop as necessary.
    """
    image_out = np.zeros(target_image_shape)
    label_out = np.zeros(target_label_shape)

    # First we pad to ensure min size is met
    to_pad = []
    for d in range(2):
        if image.shape[d + 1] < patch_size[d]:
            to_pad += [patch_size[d] - image.shape[d + 1]]
        else:
            to_pad += [0]

    pad_lb_x = to_pad[0] // 2
    pad_ub_x = to_pad[0] // 2 + to_pad[0] % 2
    pad_lb_y = to_pad[1] // 2
    pad_ub_y = to_pad[1] // 2 + to_pad[1] % 2

    # This is where we should implement any patch selection biases.
    # The final patch extracted after augmentation will always be the center of this patch
    # as this is where artefacts are least present
    crop_start_idx = []
    if len(image_properties["foreground_locations"]) == 0 or np.random.uniform() >= p_oversample_foreground:
        for d in range(2):
            if image.shape[d + 1] < patch_size[d]:
                crop_start_idx += [0]
            else:
                crop_start_idx += [np.random.randint(image.shape[d + 1] - patch_size[d] + 1)]
    else:
        location = select_foreground_voxel_to_include(image_properties)
        for d in range(2):
            if image.shape[d + 1] < patch_size[d]:
                crop_start_idx += [0]
            else:
                crop_start_idx += [
                    np.random.randint(
                        max(0, location[d] - patch_size[d]),
                        min(location[d], image.shape[d + 1] - patch_size[d]) + 1,
                    )
                ]

    image_out[:, :, :] = np.pad(
        image[
            :,
            crop_start_idx[0] : crop_start_idx[0] + patch_size[0],
            crop_start_idx[1] : crop_start_idx[1] + patch_size[1],
        ],
        ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y)),
        **pad_kwargs,
    )

    if label is None:  # Reconstruction/inpainting
        return image_out, None

    if len(label.shape) == 1:  # Classification
        return image_out, label

    label_out[:, :, :] = np.pad(
        label[
            :,
            crop_start_idx[0] : crop_start_idx[0] + patch_size[0],
            crop_start_idx[1] : crop_start_idx[1] + patch_size[1],
        ],
        ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y)),
    )

    return image_out, label_out


def select_foreground_voxel_to_include(image_properties):
    if isinstance(image_properties["foreground_locations"], list):
        locidx = np.random.choice(len(image_properties["foreground_locations"]))
        location = image_properties["foreground_locations"][locidx]
    elif isinstance(image_properties["foreground_locations"], dict):
        selected_class = np.random.choice(list(image_properties["foreground_locations"].keys()))
        locidx = np.random.choice(len(image_properties["foreground_locations"][selected_class]))
        location = image_properties["foreground_locations"][selected_class][locidx]
    return location
