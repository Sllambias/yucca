import nibabel as nib
import shutil
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p as ensure_dir_exists, subfiles
from yucca.pipeline.task_conversion.utils import generate_dataset_json
from yucca.paths import get_raw_data_path, get_source_path
from yucca.functional.testing.data.nifti import verify_spacing_is_equal, verify_orientation_is_equal


def convert(path: str = get_source_path(), subdir: str = "decathlon", subsubdir: str = "Task06_Lung"):
    # INPUT DATA
    path = f"{path}/{subdir}/{subsubdir}"
    file_suffix = ".nii.gz"

    # OUTPUT DATA
    # Define the task name and prefix
    task_name = "Task036_MSD_Lung"

    # Set target paths
    target_base = join(get_raw_data_path(), task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")

    ensure_dir_exists(target_imagesTr)
    ensure_dir_exists(target_labelsTs)
    ensure_dir_exists(target_imagesTs)
    ensure_dir_exists(target_labelsTr)

    # Split data
    images_dir_tr = join(path, "imagesTr")
    labels_dir_tr = join(path, "labelsTr")
    images_dir_ts = join(path, "imagesTs")

    # Populate Target Directory
    # This is also the place to apply any re-orientation, resampling and/or label correction.

    for sTr in subfiles(images_dir_tr, join=False):
        image_path = join(images_dir_tr, sTr)
        label_path = join(labels_dir_tr, sTr)
        sTr = sTr[: -len(file_suffix)]

        image = nib.load(image_path)
        label = nib.load(label_path)
        assert verify_spacing_is_equal(image, label), "spacing"
        assert verify_orientation_is_equal(image, label), "orientation"

        shutil.copy2(image_path, f"{target_imagesTr}/{sTr}_000.nii.gz")
        shutil.copy2(label_path, f"{target_labelsTr}/{sTr}.nii.gz")

    for sTs in subfiles(images_dir_ts, join=False):
        image_path = join(images_dir_ts, sTs)
        sTs = sTs[: -len(file_suffix)]

        image = nib.load(image_path)

        shutil.copy2(image_path, f"{target_imagesTs}/{sTs}_000.nii.gz")

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        modalities=("CT",),
        labels={0: "Background", 1: "Cancer"},
        dataset_name=task_name,
        license="CC-BY-SA 4.0",
        dataset_description="Decathlon: Lung and cancer segmentation",
        dataset_reference="The Cancer Imaging Archive",
    )
