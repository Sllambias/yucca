import yaml
import os
import nibabel as nib
import numpy as np
from PIL import Image


def load_yaml(file: str):
    with open(file, "r") as f:
        a = yaml.load(f, Loader=yaml.BaseLoader)
    return a


def read_file_to_nifti_or_np(imagepath, dtype=np.float32):
    ext = imagepath.split(os.extsep, 1)[1]
    if ext in ["nii", "nii.gz"]:
        return nib.load(imagepath)
    elif ext in ["png", "jpg", "jpeg"]:
        return np.array(Image.open(imagepath).convert("L"), dtype=dtype)
    elif ext in ["csv", "txt"]:
        return np.atleast_1d(np.genfromtxt(imagepath, delimiter=",", dtype=dtype))
    else:
        raise TypeError(f"File type invalid. Found extension: {ext} and expected one in [nii, nii.gz, png, csv, txt]")
