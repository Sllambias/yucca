from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data
from tqdm import tqdm
import nibabel as nib


def convert(path: str, subdir: str = "decathlon/Task01_BrainTumour"):
    """INPUT DATA - Define input path and suffixes"""
    path = join(path, subdir)
    ext = ".nii.gz"

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task206_BrainTumour"
    task_prefix = "BrainTumour"

    """ Access the input data. If images are not split into train/test, and you wish to randomly
    split the data, uncomment and adapt the following lines to fit your local path. """

    # NOTE: We use the test set for pre-training, as the labels are no longer available, and we thus cannot use it for evaluation!
    subjects_dir = join(path, "imagesTs")
    target_base = join(yucca_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")

    maybe_mkdir_p(target_imagesTr)

    """Populate Target Directory
    This is also the place to apply any re-orientation, resampling and/or label correction."""

    for file in tqdm(subfiles(subjects_dir, join=False, suffix=ext)):
        image_path = join(subjects_dir, file)
        file_name = file[6 : -len(ext)]  # remove the prefix BRATS_ and the suffix .nii.gz

        vol = nib.load(image_path)

        flair = vol.slicer[:, :, :, 0]
        t1w = vol.slicer[:, :, :, 1]
        t1gd = vol.slicer[:, :, :, 2]
        t2w = vol.slicer[:, :, :, 3]

        nib.save(flair, filename=f"{target_imagesTr}/{task_prefix}_{file_name}_flair_000.nii.gz")
        nib.save(t1w, filename=f"{target_imagesTr}/{task_prefix}_{file_name}_t1w_000.nii.gz")
        nib.save(t1gd, filename=f"{target_imagesTr}/{task_prefix}_{file_name}_t1gd_000.nii.gz")
        nib.save(t2w, filename=f"{target_imagesTr}/{task_prefix}_{file_name}_t2w_000.nii.gz")

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
