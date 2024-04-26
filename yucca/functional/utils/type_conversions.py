import nibabel as nib
import numpy as np
from PIL import Image
from typing import Union


def np_to_nifti_with_empty_header(array):
    image_file = nib.Nifti1Image(array, affine=None)
    image_file.set_sform(affine=np.eye(4), code=0)
    image_file.set_qform(affine=np.eye(4), code=0)
    return image_file


def nifti_or_np_to_np(array: Union[np.ndarray, nib.Nifti1Image]) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    if isinstance(array, nib.Nifti1Image):
        return array.get_fdata().astype(np.float32)
    else:
        raise TypeError(f"File data type invalid. Found: {type(array)} and expected nib.Nifti1Image or np.ndarray")


def png_to_nifti(path, maybe_to_grayscale=False, is_seg=False, to_label: int = None):
    """Takes a .png and converts it to a Nifti image using Numpy and Nibabel.
    It manually sets both sform and qform codes to 0 (which means they are invalid).
    Therefore, functions relying on trustworthy header information should discard these samples.
    Spacing is set to 1.

    if maybe_to_grayscale = True, it will check if image contains three bands, i.e. if it has color
    channels. If this is the case, it will convert to 'L' which == grayscale
    """
    dtype = np.float32
    if is_seg:
        dtype = np.uint16

    image_file = Image.open(path)
    assert len(image_file.getbands()) in [1, 3], (
        f"unexpected bands in path: {path} " "This can potentially be the alpha band. " "Implement .convert('LA') for this."
    )
    if maybe_to_grayscale and len(image_file.getbands()) == 3:
        image_file = image_file.convert("L")
    image_file = np.array(image_file, dtype=dtype)
    if is_seg and to_label:
        image_file[image_file != 0] = to_label
    image_file = nib.Nifti1Image(image_file, affine=None)
    image_file.set_sform(affine=np.eye(4), code=0)
    image_file.set_qform(affine=np.eye(4), code=0)
    return image_file
