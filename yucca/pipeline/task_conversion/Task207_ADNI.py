import gzip
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p as ensure_dir_exists, subfiles
from yucca.pipeline.task_conversion.utils import generate_dataset_json, dirs_in_dir, should_use_volume
from yucca.paths import get_raw_data_path
from tqdm import tqdm
import shutil

import nibabel as nib


def convert(path: str, subdir: str = "ADNI_NIFTI"):
    """INPUT DATA - Define input path and suffixes"""

    print("Converting task207 with try except")

    path = join(path, subdir)
    ext = ".nii"

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task207_ADNI"
    task_prefix = "ADNI"

    """ Access the input data. If images are not split into train/test, and you wish to randomly
    split the data, uncomment and adapt the following lines to fit your local path. """

    subjects_dir = join(path, "ADNI")
    target_base = join(get_raw_data_path(), task_name)
    target_imagesTr = join(target_base, "imagesTr")

    ensure_dir_exists(target_imagesTr)

    """Populate Target Directory
    This is also the place to apply any re-orientation, resampling and/or label correction."""

    skipped = []
    errors = []

    for subject in tqdm(dirs_in_dir(subjects_dir), desc="Subjects"):
        subject_dir = join(subjects_dir, subject)
        for modality in dirs_in_dir(subject_dir):
            modality_dir = join(subject_dir, modality)
            for session in dirs_in_dir(modality_dir):
                session_dir = join(modality_dir, session)
                for id in dirs_in_dir(session_dir):
                    id_dir = join(session_dir, id)
                    for file in subfiles(id_dir, join=False, suffix=ext):
                        image_path = join(id_dir, file)
                        other_info = (
                            file[: -len(ext)].replace(subject, "").replace(modality, "").replace(session, "").replace(id, "")
                        )
                        try:
                            vol = nib.load(image_path)
                            if should_use_volume(vol):
                                output_name = f"{task_prefix}_{subject}_{modality}_{session}_{id}_{other_info}_000.nii.gz"
                                output_path = join(target_imagesTr, output_name)
                                image_file = open(image_path, "rb")
                                shutil.copyfileobj(image_file, gzip.open(output_path, "wb"))
                            else:
                                skipped.append(image_path)
                        except:
                            errors.append(image_path)

    print("skipped volumes:", skipped)
    print("erroneous volumes", errors)

    print("num errors", len(errors), "num skipped", len(skipped))

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        imagesTs_dir=None,
        modalities=["MRI"],
        labels=None,
        dataset_name=task_name,
        license="you should check this if you dont know",
        dataset_description="ADNI",
        dataset_reference="ADNI",
    )
