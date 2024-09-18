import nibabel as nib
import numpy as np
import nibabel.orientations as nio
from yucca.functional.utils.nib_utils import get_nib_orientation, get_nib_spacing


def verify_spacing_is_equal(reference: nib.Nifti1Image, target: nib.Nifti1Image, id: str = ""):
    if np.allclose(get_nib_spacing(reference), get_nib_spacing(target)):
        return True
    else:
        print(
            f"Spacings do not match for {id}"
            f"Image is: {get_nib_spacing(reference)} while the label is {get_nib_spacing(target)}"
        )
        return False


def verify_orientation_is_equal(reference: nib.Nifti1Image, target: nib.Nifti1Image, id: str = ""):
    if get_nib_orientation(reference) == get_nib_orientation(target):
        return True
    else:
        print(
            f"Directions do not match for {id}"
            f"Image is: {get_nib_orientation(reference)} while the label is {get_nib_orientation(target)}"
        )
        return False


def verify_nifti_header_is_valid(image: nib.Nifti1Image):
    if image.get_qform(coded=True)[1] or image.get_sform(coded=True)[1]:
        return True
    else:
        return False


def verify_orientation_is_LR_PA_IS(image: nib.Nifti1Image):
    """
    Checks whether images are in the RAS/LPI domain, which corresponds to:
    X = Left/Right (left/right)
    Y = Posterior/Anterior (backwards/forwards)
    Z = Inferior/Superior (down/up)

    If this is not the case, then images should be converted to RAS (or at least some combination of LR-PA-IS)
    during task conversion as features such as resampling to a target spacing may become unreliably or incorrect
    """
    expected_orientation_code = np.array([0.0, 1.0, 2.0])  # This means LR-PA-IS
    orientation = get_nib_orientation(image)
    if np.all(nio.axcodes2ornt(orientation)[:, 0] == expected_orientation_code):
        return True
    else:
        return False
