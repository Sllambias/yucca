import numpy as np
import logging
import math
import nibabel as nib
import os
import torch
import torch.nn.functional as F
from yucca.functional.normalization import normalizer
from skimage.transform import resize
from yucca.functional.cropping_and_padding import pad_to_size, get_pad_kwargs
from yucca.functional.testing.data.nifti import verify_nifti_header_is_valid
from yucca.utils.nib_utils import (
    get_nib_spacing,
    get_nib_orientation,
    reorient_nib_image,
)
from yucca.utils.type_conversions import nifti_or_np_to_np
from yucca.utils.loading import read_file_to_nifti_or_np
from yucca.functional.transpose import transpose_case
from yucca.functional.testing.data.array import verify_array_shape_is_equal
from yucca.image_processing.objects.BoundingBox import get_bbox_for_foreground
from yucca.functional.cropping_and_padding import crop_to_box


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


def apply_nifti_preprocessing_and_return_numpy(
    images,
    original_size,
    target_orientation,
    label=None,
    include_header=False,
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
            metadata["reoriented"] = True
            metadata["original_orientation"] = get_nib_orientation(images[0])
            metadata["final_direction"] = target_orientation
            images = [
                reorient_nib_image(image, metadata["original_orientation"], metadata["final_direction"]) for image in images
            ]
            if label is not None and isinstance(label, nib.Nifti1Image):
                label = reorient_nib_image(label, metadata["original_orientation"], metadata["final_direction"])
        if include_header:
            metadata["header"] = images[0].header
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


def preprocess_case_for_inference(
    crop_to_nonzero,
    keep_aspect_ratio,
    images: list | tuple,
    intensities: list,
    normalization_scheme: list,
    patch_size: tuple,
    target_size,
    target_spacing,
    transpose_forward,
    allow_missing_modalities: bool = False,
) -> torch.Tensor:
    assert isinstance(images, (list, tuple)), "image(s) should be a list or tuple, even if only one image is passed"

    image_properties = {}
    ext = images[0][0].split(os.extsep, 1)[1] if isinstance(images[0], tuple) else images[0].split(os.extsep, 1)[1]
    images = [
        read_file_to_nifti_or_np(image[0]) if isinstance(image, tuple) else read_file_to_nifti_or_np(image) for image in images
    ]

    image_properties["image_extension"] = ext
    image_properties["original_shape"] = np.array(images[0].shape)

    assert len(image_properties["original_shape"]) in [
        2,
        3,
    ], "images must be either 2D or 3D for preprocessing"

    images, _, image_properties["nifti_metadata"] = apply_nifti_preprocessing_and_return_numpy(
        images, image_properties["original_shape"], label=None, include_header=True
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

    verify_array_shape_is_equal(reference=images.shape[2:], target=image_properties["padded_shape"])

    shape = images.shape[2:]
    if len(pad) == 6:
        images = images[
            :,
            :,
            pad[0] : shape[0] - pad[1],
            pad[2] : shape[1] - pad[3],
            pad[4] : shape[2] - pad[5],
        ]
    elif len(pad) == 4:
        images = images[:, :, pad[0] : shape[0] - pad[1], pad[2] : shape[1] - pad[3]]

    verify_array_shape_is_equal(reference=images.shape[2:], target=image_properties["resampled_transposed_shape"])

    # Here we Interpolate the array to the original size. The shape starts as [H, W (,D)]. For Torch functionality it is changed to [B, C, H, W (,D)].
    # Afterwards it's squeezed back into [H, W (,D)] and transposed to the original direction.
    images = F.interpolate(images, size=shape_after_crop_transposed.tolist(), mode="trilinear").permute(
        [0, 1] + [i + 2 for i in transpose_backward]
    )

    # Now move the tensor to the CPU
    images = images.cpu()

    verify_array_shape_is_equal(reference=images.shape[2:], target=image_properties["cropped_shape"])

    if crop_to_nonzero:
        bbox = image_properties["nonzero_box"]
        slices = [
            slice(None),
            slice(None),
            slice(bbox[0], bbox[1]),
            slice(bbox[2], bbox[3]),
        ]
        if len(bbox) == 6:
            slices.append(
                slice(bbox[4], bbox[5]),
            )
        canvas[slices] = images
    else:
        canvas = images
    return canvas.numpy(), image_properties
