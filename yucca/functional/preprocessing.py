import numpy as np
import logging
import math
import nibabel as nib
import torch
import torch.nn.functional as F
import cc3d
from typing import Union, List, Optional
from yucca.functional.array_operations.normalization import normalizer
from skimage.transform import resize
from yucca.functional.array_operations.cropping_and_padding import pad_to_size, get_pad_kwargs
from yucca.functional.testing.data.nifti import verify_nifti_header_is_valid, verify_orientation_is_LR_PA_IS
from yucca.functional.testing.data.array import verify_shape_is_equal
from yucca.functional.array_operations.transpose import transpose_case, transpose_array
from yucca.functional.array_operations.bounding_boxes import get_bbox_for_foreground
from yucca.functional.array_operations.cropping_and_padding import crop_to_box
from yucca.functional.utils.nib_utils import (
    get_nib_spacing,
    get_nib_orientation,
    reorient_nib_image,
)
from yucca.functional.utils.type_conversions import nifti_or_np_to_np


def analyze_label(label, enable_connected_components_analysis: bool = False, spacing: list = [], per_class=False):
    # we get some (no need to get all) locations of foreground, that we will later use in the
    # oversampling of foreground classes
    # And we also potentially analyze the connected components of the label
    foreground_locs = get_foreground_locations(label, per_class=per_class, max_locs_total=100000)
    if not enable_connected_components_analysis:
        label_cc_n = 0
        label_cc_sizes = 0
    else:
        numbered_ground_truth, label_cc_n = cc3d.connected_components(label, connectivity=26, return_N=True)
        if len(numbered_ground_truth) == 0:
            label_cc_sizes = 0
        else:
            label_cc_sizes = [i * np.prod(spacing) for i in np.unique(numbered_ground_truth, return_counts=True)[-1][1:]]
    return foreground_locs, label_cc_n, label_cc_sizes


def get_foreground_locations(label, per_class=False, max_locs_total=100000):
    foreground_locations = {}
    if not per_class:
        foreground_locs_for_all = np.array(np.nonzero(label)).T[::10].tolist()
        if len(foreground_locs_for_all) > 0:
            if len(foreground_locs_for_all) > max_locs_total:
                foreground_locs_for_all = foreground_locs_for_all[:: round(len(foreground_locs_for_all) / max_locs_total)]
            foreground_locations["1"] = foreground_locs_for_all
    else:
        foreground_classes_present = np.unique(label)[1:]
        if len(foreground_classes_present) == 0:
            return foreground_locations
        max_locs_per_class = int(max_locs_total / len(foreground_classes_present))
        for c in foreground_classes_present:
            foreground_locs_for_c = np.array(np.where(label == int(c))).T[::10]
            if len(foreground_locs_for_c) > 0:
                if len(foreground_locs_for_c) > max_locs_per_class:
                    foreground_locs_for_c = foreground_locs_for_c[:: round(len(foreground_locs_for_c) / max_locs_per_class)]
                foreground_locations[str(int(c))] = foreground_locs_for_c
    return foreground_locations


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
    norm_op: str,
    intensities: list = None,
    label: np.ndarray = None,
    allow_missing_modalities: bool = False,
):
    # Normalize and Transpose images to target view.
    # Transpose labels to target view.
    assert len(case) == len(norm_op), (
        "number of images and "
        "normalization  operations does not match. \n"
        f"len(images) == {len(case)} \n"
        f"len(norm_op) == {len(norm_op)} \n"
    )

    for i in range(len(case)):
        image = case[i]
        assert image is not None
        if image.size == 0:
            assert allow_missing_modalities is True, "missing modality and allow_missing_modalities is not enabled"
        else:
            # Normalize
            if intensities is not None:
                case[i] = normalizer(image, scheme=norm_op[i], intensities=intensities[i])
            else:
                case[i] = normalizer(image, scheme=norm_op[i])

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


def apply_nifti_preprocessing_and_return_numpy(
    images,
    original_size,
    target_orientation,
    label=None,
    include_header=False,
    strict=True,
):
    # If qform and sform are both missing the header is corrupt and we do not trust the
    # direction from the affine
    # Make sure you know what you're doing
    metadata = {
        "original_spacing": np.array([1.0] * len(original_size)).tolist(),
        "original_orientation": None,
        "final_direction": None,
        "header": None,
        "affine": None,
        "reoriented": False,
    }

    if isinstance(images[0], nib.Nifti1Image):
        # If qform and sform are both missing the header is corrupt and we do not trust the
        # direction from the affine
        # Make sure you know what you're doing

        if verify_nifti_header_is_valid(images[0]) is True:
            if strict:
                assert verify_orientation_is_LR_PA_IS(
                    images[0]
                ), "unexpected NIFTI axes. Consider RAS-conversion during task conversion"
            metadata["reoriented"] = True
            metadata["original_orientation"] = get_nib_orientation(images[0])
            metadata["final_direction"] = target_orientation
            images = [
                reorient_nib_image(image, metadata["original_orientation"], metadata["final_direction"]) for image in images
            ]
            if label is not None and isinstance(label, nib.Nifti1Image):
                label = reorient_nib_image(label, metadata["original_orientation"], metadata["final_direction"])
        if include_header:
            metadata["header"] = images[0].header.binaryblock
        metadata["original_spacing"] = get_nib_spacing(images[0]).tolist()
        metadata["affine"] = images[0].affine

    images = [nifti_or_np_to_np(image) for image in images]
    if label is not None:
        label = nifti_or_np_to_np(label)
    return images, label, metadata


def determine_target_size(
    images_transposed: list,
    original_spacing,
    transpose_forward,
    target_size,
    target_spacing,
    keep_aspect_ratio,
):
    image_shape_t = np.array(images_transposed[0].shape)
    original_spacing_t = original_spacing[transpose_forward]

    # We do not want to change the aspect ratio so we resample using the minimum alpha required
    # to attain 1 correct dimension, and then the rest will be padded.
    # Additionally we make sure each dimension is divisible by 16 to avoid issues with standard pooling/stride settings
    if target_size is not None:
        resample_target_size, final_target_size, new_spacing = determine_resample_size_from_target_size(
            current_size=image_shape_t,
            current_spacing=original_spacing_t,
            target_size=target_size,
            keep_aspect_ratio=keep_aspect_ratio,
        )

    # Otherwise we need to calculate a new target shape, and we need to factor in that
    # the images will first be transposed and THEN resampled.
    # Find new shape based on the target spacing
    elif target_spacing is not None:
        target_spacing = np.array(target_spacing, dtype=float)
        target_spacing_t = target_spacing[transpose_forward]
        resample_target_size, final_target_size, new_spacing = determine_resample_size_from_target_spacing(
            current_size=image_shape_t, current_spacing=original_spacing_t, target_spacing=target_spacing_t
        )
    else:
        resample_target_size = image_shape_t
        final_target_size = None
        new_spacing = original_spacing_t.tolist()
    return resample_target_size, final_target_size, new_spacing


def preprocess_case_for_training_with_label(
    images: List[Union[np.ndarray, nib.Nifti1Image]],
    label: Union[np.ndarray, nib.Nifti1Image],
    normalization_operation: list,
    allow_missing_modalities: bool = False,
    background_pixel_value: int = 0,
    enable_cc_analysis: bool = False,
    foreground_locs_per_label: bool = False,
    missing_modality_idxs: list = [],
    crop_to_nonzero: bool = True,
    keep_aspect_ratio_when_using_target_size: bool = False,
    image_properties: Optional[dict] = {},
    intensities: Optional[List] = None,
    target_orientation: Optional[str] = "RAS",
    target_size: Optional[List] = None,
    target_spacing: Optional[List] = None,
    transpose: Optional[list] = [0, 1, 2],
):
    """
    one of target_size or target_spacing is required.
    """

    images, label, image_properties["nifti_metadata"] = apply_nifti_preprocessing_and_return_numpy(
        images=images,
        original_size=np.array(images[0].shape),
        target_orientation=target_orientation,
        label=label,
        include_header=False,
    )

    original_size = images[0].shape

    # Cropping is performed to save computational resources. We are only removing background.
    if crop_to_nonzero:
        nonzero_box = get_bbox_for_foreground(images[0], background_label=background_pixel_value)
        image_properties["crop_to_nonzero"] = nonzero_box
        for i in range(len(images)):
            images[i] = crop_to_box(images[i], nonzero_box)
        label = crop_to_box(label, nonzero_box)
    else:
        image_properties["crop_to_nonzero"] = crop_to_nonzero

    image_properties["size_before_transpose"] = list(images[0].shape)

    images = transpose_case(images, axes=transpose)
    label = transpose_array(label, axes=transpose)

    image_properties["size_after_transpose"] = list(images[0].shape)

    resample_target_size, final_target_size, new_spacing = determine_target_size(
        images_transposed=images,
        original_spacing=np.array(image_properties["nifti_metadata"]["original_spacing"]),
        transpose_forward=transpose,
        target_size=target_size,
        target_spacing=target_spacing,
        keep_aspect_ratio=keep_aspect_ratio_when_using_target_size,
    )

    # here we need to make sure missing modalities are accounted for, as the order of the images
    # can matter for normalization operations
    for missing_mod in missing_modality_idxs:
        images.insert(missing_mod, np.array([]))
    images, label = resample_and_normalize_case(
        case=images,
        target_size=resample_target_size,
        norm_op=normalization_operation,
        intensities=intensities,
        label=label,
        allow_missing_modalities=allow_missing_modalities,
    )

    if final_target_size is not None:
        images, label = pad_case_to_size(case=images, size=final_target_size, label=label)
    (
        image_properties["foreground_locations"],
        image_properties["label_cc_n"],
        image_properties["label_cc_sizes"],
    ) = analyze_label(
        label=label, enable_connected_components_analysis=enable_cc_analysis, per_class=foreground_locs_per_label
    )

    first_existing_modality = list(set(range(len(images))).difference(missing_modality_idxs))[0]
    image_properties["new_size"] = list(images[first_existing_modality].shape)
    image_properties["original_spacing"] = image_properties["nifti_metadata"]["original_spacing"]
    image_properties["original_size"] = original_size
    image_properties["original_orientation"] = image_properties["nifti_metadata"]["original_orientation"]
    image_properties["new_spacing"] = new_spacing
    image_properties["new_direction"] = image_properties["nifti_metadata"]["final_direction"]
    return images, label, image_properties


def preprocess_case_for_training_without_label(
    images: List[Union[np.ndarray, nib.Nifti1Image]],
    normalization_operation: list,
    allow_missing_modalities: bool = False,
    background_pixel_value: int = 0,
    missing_modality_idxs: list = [],
    crop_to_nonzero: bool = True,
    keep_aspect_ratio_when_using_target_size: bool = False,
    image_properties: Optional[dict] = {},
    intensities: Optional[List] = None,
    target_orientation: Optional[str] = "RAS",
    target_size: Optional[List] = None,
    target_spacing: Optional[List] = None,
    transpose: Optional[list] = [0, 1, 2],
    strict: bool = True,
):
    """
    one of target_size or target_spacing is required.
    """

    images, label, image_properties["nifti_metadata"] = apply_nifti_preprocessing_and_return_numpy(
        images=images,
        original_size=np.array(images[0].shape),
        target_orientation=target_orientation,
        label=None,
        include_header=False,
        strict=strict,
    )

    original_size = images[0].shape

    # Cropping is performed to save computational resources. We are only removing background.
    if crop_to_nonzero:
        nonzero_box = get_bbox_for_foreground(images[0], background_label=background_pixel_value)
        image_properties["crop_to_nonzero"] = nonzero_box
        for i in range(len(images)):
            images[i] = crop_to_box(images[i], nonzero_box)
    else:
        image_properties["crop_to_nonzero"] = crop_to_nonzero

    image_properties["size_before_transpose"] = list(images[0].shape)

    images = transpose_case(images, axes=transpose)

    image_properties["size_after_transpose"] = list(images[0].shape)

    resample_target_size, final_target_size, new_spacing = determine_target_size(
        images_transposed=images,
        original_spacing=np.array(image_properties["nifti_metadata"]["original_spacing"]),
        transpose_forward=transpose,
        target_size=target_size,
        target_spacing=target_spacing,
        keep_aspect_ratio=keep_aspect_ratio_when_using_target_size,
    )

    # here we need to make sure missing modalities are accounted for, as the order of the images
    # can matter for normalization operations
    for missing_mod in missing_modality_idxs:
        images.insert(missing_mod, np.array([]))

    images = resample_and_normalize_case(
        case=images,
        target_size=resample_target_size,
        norm_op=normalization_operation,
        intensities=intensities,
        allow_missing_modalities=allow_missing_modalities,
    )

    if final_target_size is not None:
        images = pad_case_to_size(case=images, size=final_target_size, label=None)

    first_existing_modality = list(set(range(len(images))).difference(missing_modality_idxs))[0]
    image_properties["new_size"] = list(images[first_existing_modality].shape)
    image_properties["label_cc_n"] = image_properties["label_cc_sizes"] = 0
    image_properties["foreground_locations"] = []
    image_properties["original_spacing"] = image_properties["nifti_metadata"]["original_spacing"]
    image_properties["original_size"] = original_size
    image_properties["original_orientation"] = image_properties["nifti_metadata"]["original_orientation"]
    image_properties["new_spacing"] = new_spacing
    image_properties["new_direction"] = image_properties["nifti_metadata"]["final_direction"]
    return images, image_properties


def preprocess_case_for_inference(
    crop_to_nonzero,
    images: list | tuple,
    intensities: list,
    normalization_scheme: list,
    patch_size: Union[tuple, None],
    target_size,
    target_spacing,
    target_orientation,
    allow_missing_modalities: bool = False,
    ext=".nii.gz",
    keep_aspect_ratio: bool = True,
    transpose_forward=[0, 1, 2],
) -> torch.Tensor:
    assert isinstance(images, (list, tuple)), "image(s) should be a list or tuple, even if only one image is passed"

    image_properties = {}
    image_properties["image_extension"] = ext
    image_properties["original_shape"] = np.array(images[0].shape)

    assert len(image_properties["original_shape"]) in [
        2,
        3,
    ], "images must be either 2D or 3D for preprocessing"

    images, _, image_properties["nifti_metadata"] = apply_nifti_preprocessing_and_return_numpy(
        images, image_properties["original_shape"], target_orientation=target_orientation, label=None, include_header=True
    )

    image_properties["uncropped_shape"] = np.array(images[0].shape)

    if crop_to_nonzero:
        nonzero_box = get_bbox_for_foreground(images[0], background_label=0)
        for i in range(len(images)):
            images[i] = crop_to_box(images[i], nonzero_box)
        image_properties["nonzero_box"] = nonzero_box

    image_properties["cropped_shape"] = np.array(images[0].shape)

    images = transpose_case(images, axes=transpose_forward)

    resample_target_size, _, _ = determine_target_size(
        images_transposed=images,
        original_spacing=np.array(image_properties["nifti_metadata"]["original_spacing"]),
        transpose_forward=transpose_forward,
        target_size=target_size,
        target_spacing=target_spacing,
        keep_aspect_ratio=keep_aspect_ratio,
    )

    images = resample_and_normalize_case(
        case=images,
        target_size=resample_target_size,
        norm_op=normalization_scheme,
        intensities=intensities,
        label=None,
        allow_missing_modalities=allow_missing_modalities,
    )

    # From this point images are shape (1, c, x, y, z)
    image_properties["resampled_transposed_shape"] = np.array(images[0].shape)

    for i in range(len(images)):
        images[i], padding = pad_to_size(images[i], patch_size)

    image_properties["padded_shape"] = np.array(images[0].shape)
    image_properties["padding"] = padding

    # Stack and fix dimensions
    images = np.stack(images)[np.newaxis]

    return torch.tensor(images, dtype=torch.float32), image_properties


def reverse_preprocessing(crop_to_nonzero, images, image_properties, n_classes, transpose_forward, transpose_backward):
    """
    Expects images to be preprocessed by the "preprocess_case_for_inference" function or similar that
    will produce an image_properties dict with instructions on how to reverse the operations.

    Expected shape of images are:
    (b, c, x, y(, z))

    (1) Initialization: Extract relevant properties from the image_properties dictionary.
    (2) Padding Reversion: Reverse the padding applied during preprocessing.
    (3) Resampling and Transposition Reversion: Resize the images to revert the resampling operation.
    Transpose the images back to the original orientation.
    (4) Cropping Reversion (Optional): If cropping to the nonzero bounding box was applied, revert the cropping operation.
    (5) Return: Return the reverted images as a NumPy array.
    The original orientation of the image will be re-applied when saving the prediction
    """
    image_properties["save_format"] = image_properties.get("image_extension")
    canvas = torch.zeros((1, n_classes, *image_properties["uncropped_shape"]), dtype=images.dtype)
    shape_after_crop = image_properties["cropped_shape"]
    shape_after_crop_transposed = shape_after_crop[transpose_forward]
    pad = image_properties["padding"]

    verify_shape_is_equal(reference=images.shape[2:], target=image_properties["padded_shape"])

    shape = images.shape[2:]
    if len(pad) == 6:
        images = images[
            :,
            :,
            pad[0] : shape[0] - pad[1],
            pad[2] : shape[1] - pad[3],
            pad[4] : shape[2] - pad[5],
        ]
        mode = "trilinear"
    elif len(pad) == 4:
        images = images[:, :, pad[0] : shape[0] - pad[1], pad[2] : shape[1] - pad[3]]
        mode = "bilinear"

    verify_shape_is_equal(reference=images.shape[2:], target=image_properties["resampled_transposed_shape"])

    # Here we Interpolate the array to the original size. The shape starts as [H, W (,D)]. For Torch functionality it is changed to [B, C, H, W (,D)].
    # Afterwards it's squeezed back into [H, W (,D)] and transposed to the original direction.
    images = F.interpolate(images, size=shape_after_crop_transposed.tolist(), mode=mode).permute(
        [0, 1] + [i + 2 for i in transpose_backward]
    )

    # Now move the tensor to the CPU
    images = images.cpu()

    verify_shape_is_equal(reference=images.shape[2:], target=image_properties["cropped_shape"])

    if crop_to_nonzero:
        bbox = image_properties["nonzero_box"]
        slices = [
            slice(None),
            slice(None),
            slice(bbox[0], bbox[1] + 1),
            slice(bbox[2], bbox[3] + 1),
        ]
        if len(bbox) == 6:
            slices.append(
                slice(bbox[4], bbox[5] + 1),
            )
        canvas[slices] = images
    else:
        canvas = images
    return canvas.numpy(), image_properties
