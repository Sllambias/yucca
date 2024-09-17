from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p as ensure_dir_exists, subfiles
from yucca.pipeline.task_conversion.utils import generate_dataset_json, dirs_in_dir
from yucca.paths import get_raw_data_path
from tqdm import tqdm
import shutil


def convert(path: str, subdir: str = "ISLES-2022"):
    """INPUT DATA - Define input path and suffixes"""
    path = join(path, subdir)
    ext = ".nii.gz"

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task202_ISLES22"
    task_prefix = "ISLES22"

    """ Access the input data. If images are not split into train/test, and you wish to randomly
    split the data, uncomment and adapt the following lines to fit your local path. """

    subjects_dir = join(path, "images")
    target_base = join(get_raw_data_path(), task_name)
    target_imagesTr = join(target_base, "imagesTr")

    ensure_dir_exists(target_imagesTr)

    """Populate Target Directory
    This is also the place to apply any re-orientation, resampling and/or label correction."""

    for subject in tqdm(dirs_in_dir(subjects_dir), desc="Subjects"):
        subject_dir = join(subjects_dir, subject)
        for session in dirs_in_dir(subject_dir):
            session_dir = join(subject_dir, session)
            for modality in dirs_in_dir(session_dir):
                modality_dir = join(session_dir, modality)
                for file in subfiles(modality_dir, join=False, suffix=ext):
                    image_path = join(modality_dir, file)
                    other_info = file[: -len(ext)].replace(subject, "").replace(session, "").replace(modality, "")
                    output_name = f"{task_prefix}_{subject}_{modality}_{session}_{other_info}_000.nii.gz"
                    output_path = join(target_imagesTr, output_name)
                    shutil.copy2(image_path, output_path)

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        imagesTs_dir=None,
        modalities=["MRI"],
        labels=None,
        dataset_name=task_name,
        license="CC-BY-SA 4.0",
        dataset_description="Decathlon: Brain Tumour Segmentation",
        dataset_reference="King's College London",
    )
