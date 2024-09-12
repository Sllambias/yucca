import shutil
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p as ensure_dir_exists, subfiles
from yucca.pipeline.task_conversion.utils import generate_dataset_json, should_use_volume, dirs_in_dir
from yucca.paths import get_raw_data_path
from tqdm import tqdm
import nibabel as nib


rejec_modalities = ["fmri", "func"]  # TODO: Might need to skip GRE


def should_skip(modality: str):
    modality = modality.lower()
    for reject in rejec_modalities:
        if reject in modality:
            return True
    return False


def convert(path: str, subdir: str = "OASIS4"):
    """INPUT DATA - Define input path and suffixes"""
    path = join(path, subdir)

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task204_OASIS4"
    task_prefix = "OASIS4"

    subjects_dir = join(path, "OASIS4_images")
    target_base = join(get_raw_data_path(), task_name)
    target_imagesTr = join(target_base, "imagesTr")

    ensure_dir_exists(target_imagesTr)

    """Populate Target Directory"""

    ext = ".nii.gz"

    skipped_volumes = []
    skipped_modalities = []
    for subject in tqdm(dirs_in_dir(subjects_dir), desc="Subjects"):
        if "mr" not in subject.lower():
            continue  # skip this subject

        subject_dir = join(subjects_dir, subject)
        for modality in dirs_in_dir(subject_dir):
            if should_skip(modality):
                skipped_modalities.append(modality)
                continue  # skip this modality
            modality_dir = join(subject_dir, modality, "NIFTI")
            for file in subfiles(modality_dir, join=False, suffix=ext):
                image_path = join(modality_dir, file)
                other_info = file[: -len(ext)].replace(subject, "").replace(modality, "")
                vol = nib.load(image_path)
                if should_use_volume(vol):
                    output_name = f"{task_prefix}_{subject}_{modality}_{other_info}_000.nii.gz"
                    print(output_name)
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
        labels={},
        dataset_name=task_name,
        license="https://www.oasis-brains.org/#access",
        dataset_description="Open Access Series of Imaging Studies (OASIS)",
        dataset_reference="https://www.oasis-brains.org",
    )
