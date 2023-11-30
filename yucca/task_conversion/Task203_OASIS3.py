import shutil
import gzip
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data
from datetime import datetime
from tqdm import tqdm

import nibabel as nib
import numpy as np


accept_modalities = ["t1", "t2", "mprage", "flair", "gre", "dwi", "swi", "grappa", "anat"]
rejec_modalities = ["fmri", "func"]


def should_skip(modality: str):
    modality = modality.lower()
    for reject in rejec_modalities:
        if reject in modality:
            return True
    for accept in accept_modalities:
        if accept in modality:
            return False
    return True


def should_use(vol: nib.Nifti1Image):
    if np.any(np.array(vol.shape) < 15):
        return False
    return True


def dirs_in_dir(dir: str):
    p = Path(dir)
    return [f.name for f in p.iterdir() if f.is_dir() and f.name[0] not in [".", "_"]]


def convert(path: str, subdir: str = "OASIS3"):
    """INPUT DATA - Define input path and suffixes"""
    path = join(path, subdir)

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task203_OASIS3"
    task_prefix = "OASIS3"

    """ Access the input data. If images are not split into train/test, and you wish to randomly
    split the data, uncomment and adapt the following lines to fit your local path. """

    subjects_dir = join(path, "DATA")
    target_base = join(yucca_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")

    maybe_mkdir_p(target_imagesTr)

    """Populate Target Directory
    This is also the place to apply any re-orientation, resampling and/or label correction."""

    ext = ".nii.gz"

    for subject in tqdm(dirs_in_dir(subjects_dir), desc="Subjects"):
        if "mr" not in subject:
            continue  # skip this subject

        subject_dir = join(subjects_dir, subject)
        for modality in dirs_in_dir(subject_dir):
            modality_dir = join(subject_dir, modality)
            if should_skip(modality):
                continue  # skip this modality
            for file in subfiles(modality_dir, join=False, suffix=ext):
                image_path = join(modality_dir, file)
                file_name = file[: -len(ext)]
                vol = nib.load(image_path)
                if should_use(vol):
                    output_name = f"{task_prefix}_{file_name}_000.nii.gz"
                    output_path = join(target_imagesTr, output_name)
                    with open(image_path, "rb") as f_in:
                        with gzip.open(output_path, mode="wb", compresslevel=1) as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    print("Volume not large enough", image_path)

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        imagesTs_dir=None,
        modalities=accept_modalities,
        labels=None,
        dataset_name=task_name,
        license="https://www.oasis-brains.org/#access",
        dataset_description="Open Access Series of Imaging Studies (OASIS)",
        dataset_reference="https://www.oasis-brains.org",
    )
