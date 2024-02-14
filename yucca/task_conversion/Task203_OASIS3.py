import shutil
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.task_conversion.utils import generate_dataset_json, dirs_in_dir, should_use_volume
from yucca.paths import yucca_raw_data
from tqdm import tqdm
import nibabel as nib


accept_modalities = ["t1", "t2", "mprage", "flair", "gre", "dwi", "swi", "grappa", "anat"]
rejec_modalities = ["fmri", "func"]  # TODO: Might need to skip GRE


def should_skip(modality: str):
    modality = modality.lower()
    for reject in rejec_modalities:
        if reject in modality:
            return True
    for accept in accept_modalities:
        if accept in modality:
            return False
    return True


def convert(path: str, subdir: str = "OASIS3"):
    """INPUT DATA - Define input path and suffixes"""
    path = join(path, subdir)

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task203_OASIS3"
    task_prefix = "OASIS3"

    subjects_dir = join(path, "DATA")
    target_base = join(yucca_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")

    maybe_mkdir_p(target_imagesTr)

    """Populate Target Directory"""

    ext = ".nii.gz"

    skipped_volumes = []
    skipped_modalities = []

    for subject in tqdm(dirs_in_dir(subjects_dir), desc="Subjects"):
        if "mr" not in subject.lower():
            continue  # skip this subject

        subject_dir = join(subjects_dir, subject)
        for modality in dirs_in_dir(subject_dir):
            modality_dir = join(subject_dir, modality)
            if should_skip(modality):
                skipped_modalities.append(modality)
                continue  # skip this modality
            for file in subfiles(modality_dir, join=False, suffix=ext):
                image_path = join(modality_dir, file)
                file_name = file[: -len(ext)]
                vol = nib.load(image_path)
                if should_use_volume(vol):
                    output_name = f"{task_prefix}_{file_name}_000.nii.gz"
                    output_path = join(target_imagesTr, output_name)
                    shutil.copy2(image_path, output_path)
                else:
                    skipped_volumes.append(image_path)

    print(f"Skipped {len(skipped_volumes)} and {len(skipped_modalities)}")
    print("Skipped workers", skipped_volumes)
    print("Skipped modalities", skipped_modalities)

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        imagesTs_dir=None,
        modalities=["MRI"],
        labels=None,
        dataset_name=task_name,
        license="https://www.oasis-brains.org/#access",
        dataset_description="Open Access Series of Imaging Studies (OASIS)",
        dataset_reference="https://www.oasis-brains.org",
    )
