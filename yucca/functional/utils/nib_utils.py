import nibabel.orientations as nio
import numpy as np
import nibabel as nib


def get_nib_spacing(nib_image: nib.Nifti1Image) -> np.ndarray:
    return np.array(nib_image.header.get_zooms())


def get_nib_orientation(nib_image: nib.Nifti1Image) -> str:
    affine = nib_image.affine
    return "".join(nio.aff2axcodes(affine))


def reorient_nib_image(nib_image, original_orientation: str, target_orientation: str) -> np.ndarray:
    # The reason we don't use the affine information to get original_orientation is that it can be
    # incorrect. Therefore it can be manually specified. In the cases where header can be trusted,
    # Just use get_nib_orientation to get the original_orientation.
    if original_orientation == target_orientation:
        return nib_image
    start = nio.axcodes2ornt(original_orientation)
    end = nio.axcodes2ornt(target_orientation)
    orientation = nio.ornt_transform(start, end)
    return nib_image.as_reoriented(orientation)
