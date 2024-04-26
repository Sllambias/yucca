import nibabel as nib
import numpy as np
from yucca.functional.utils.nib_utils import get_nib_orientation, get_nib_spacing


def verify_spacing_is_equal(reference: nib.Nifti1Image, target: nib.Nifti1Image, id: str = ""):
    assert np.allclose(get_nib_spacing(reference), get_nib_spacing(target)), (
        f"Spacings do not match for {id}"
        f"Image is: {get_nib_spacing(reference)} while the label is {get_nib_spacing(target)}"
    )


def verify_orientation_is_equal(reference: nib.Nifti1Image, target: nib.Nifti1Image, id: str = ""):
    assert get_nib_orientation(reference) == get_nib_orientation(target), (
        f"Directions do not match for {id}"
        f"Image is: {get_nib_orientation(reference)} while the label is {get_nib_orientation(target)}"
    )


def verify_nifti_header_is_valid(image: nib.Nifti1Image):
    if image.get_qform(coded=True)[1] or image.get_sform(coded=True)[1]:
        return True
    else:
        return False
