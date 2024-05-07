import shutil
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.task_conversion.utils import generate_dataset_json, should_use_volume
from yucca.paths import yucca_raw_data
from tqdm import tqdm

import nibabel as nib


def convert(path: str, subdir: str = "decathlon/Task04_Hippocampus"):
    """INPUT DATA - Define input path and suffixes"""
    path = join(path, subdir)
    ext = ".nii.gz"

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task205_Hippocampus"
    task_prefix = "Hippocampus"

    """ Access the input data. If images are not split into train/test, and you wish to randomly
    split the data, uncomment and adapt the following lines to fit your local path. """

    subjects_dir = join(path, "imagesTr")
    target_base = join(yucca_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")

    maybe_mkdir_p(target_imagesTr)

    """Populate Target Directory
    This is also the place to apply any re-orientation, resampling and/or label correction."""

    skipped_volumes = []

    for file in tqdm(subfiles(subjects_dir, join=False, suffix=ext)):
        image_path = join(subjects_dir, file)
        file_name = file[: -len(ext)]
        vol = nib.load(image_path)
        if should_use_volume(vol):
            output_name = f"{task_prefix}_{file_name}_000.nii.gz"
            output_path = join(target_imagesTr, output_name)
            shutil.copy2(image_path, output_path)
        else:
            skipped_volumes.append(image_path)

    print(f"Skipped {len(skipped_volumes)}")
    print("Skipped volumes", skipped_volumes)

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        imagesTs_dir=None,
        modalities=["MRI"],
        labels=None,
        dataset_name=task_name,
        license="CC-BY-SA 4.0",
        dataset_description="Decathlon: Left and right hippocampus segmentation",
        dataset_reference="Vanderbilt University Medical Center",
    )
