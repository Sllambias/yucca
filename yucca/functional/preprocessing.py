import numpy as np
import logging
import math
from yucca.preprocessing.normalization import normalizer
from skimage.transform import resize
from yucca.image_processing.cropping_and_padding import pad_to_size, get_pad_kwargs


def determine_resample_size_from_target_size(current_size, current_spacing, target_size, keep_aspect_ratio: bool = False):
    if keep_aspect_ratio:
        resample_target_size = np.array(current_size * np.min(target_size / current_size)).astype(int)
        final_target_size = target_size
        final_target_size = [math.ceil(i / 16) * 16 for i in final_target_size]
    else:
        resample_target_size = target_size
        resample_target_size = [math.ceil(i / 16) * 16 for i in resample_target_size]
        final_target_size = None
    new_spacing = (
        (np.array(resample_target_size).astype(float) / current_size.astype(float)) * np.array(current_spacing).astype(float)
    ).tolist()
    return resample_target_size, final_target_size, new_spacing


def determine_resample_size_from_target_spacing(current_size, current_spacing, target_spacing: np.ndarray):
    final_target_size = None
    resample_target_size = np.round((current_spacing / target_spacing).astype(float) * current_size).astype(int)
    new_spacing = target_spacing.tolist()
    return resample_target_size, final_target_size, new_spacing


def resample_and_normalize_case(
    case: list,
    target_size,
    intensities: list,
    norm_op: str,
    label: np.ndarray = None,
    allow_missing_modalities: bool = False,
):
    # Normalize and Transpose images to target view.
    # Transpose labels to target view.
    assert len(case) == len(norm_op) == len(intensities), (
        "number of images, "
        "normalization  operations and intensities does not match. \n"
        f"len(images) == {len(case)} \n"
        f"len(norm_op) == {len(norm_op)} \n"
        f"len(intensities) == {len(intensities)} \n"
    )

    for i in range(len(case)):
        image = case[i]
        assert image is not None
        if image.size == 0:
            assert allow_missing_modalities is True, "missing modality and allow_missing_modalities is not enabled"
        else:
            # Normalize
            case[i] = normalizer(image, scheme=norm_op[i], intensities=intensities[i])
            # Resample to target shape and spacing
            try:
                case[i] = resize(case[i], output_shape=target_size, order=3)
            except OverflowError:
                logging.error("Unexpected values in either shape or image for resize")
    if label is not None:
        try:
            label = resize(label, output_shape=target_size, order=0, anti_aliasing=False)
        except OverflowError:
            logging.error("Unexpected values in either shape or label for resize")
        return case, label
    return case


def pad_case_to_size(case: list, size, pad_value="min", label=None):
    for i in range(len(case)):
        pad_kwargs = get_pad_kwargs(data=case[i], pad_value=pad_value)
        case[i], _ = pad_to_size(case[i], size, **pad_kwargs)
    if label is None:
        return case
    label, _ = pad_to_size(label, size, **pad_kwargs)
    return case, label
