import shutil
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p as ensure_dir_exists, subfiles
from yucca.task_conversion.utils import generate_dataset_json, remove_punctuation_and_spaces
from yucca.paths import get_raw_data_path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def convert(path: str, subdir: str = "Autopet"):
    """INPUT DATA - Define input path and suffixes"""
    path = join(path, subdir)

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task013_AutoPET3"
    task_prefix = "AutoPET3"

    # Target paths
    target_base = join(get_raw_data_path(), task_name)

    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")

    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")

    ensure_dir_exists(target_imagesTr)
    ensure_dir_exists(target_labelsTs)
    ensure_dir_exists(target_imagesTs)
    ensure_dir_exists(target_labelsTr)

    # Input paths
    images_dir = join(path, "imagesTr")
    labels_dir = join(path, "labelsTr")

    training_samples, test_samples = train_test_split(subfiles(labels_dir, join=False), random_state=859032)

    ###Populate Target Directory###
    for sTr in tqdm(training_samples, desc="Training cases"):
        sTr = sTr[: -len(".nii.gz")]
        sTr_valid = remove_punctuation_and_spaces(sTr)

        # Loading relevant modalities and the ground truth
        src_image_file_path1 = join(images_dir, sTr + "_0000.nii.gz")
        src_image_file_path2 = join(images_dir, sTr + "_0001.nii.gz")
        src_label_path = join(labels_dir, sTr + ".nii.gz")

        dst_image_file_path1 = f"{target_imagesTr}/{task_prefix}_{sTr_valid}_000.nii.gz"
        dst_image_file_path2 = f"{target_imagesTr}/{task_prefix}_{sTr_valid}_001.nii.gz"
        dst_label_path = f"{target_labelsTr}/{task_prefix}_{sTr_valid}.nii.gz"

        shutil.copy2(src_image_file_path1, dst_image_file_path1)
        shutil.copy2(src_image_file_path2, dst_image_file_path2)
        shutil.copy2(src_label_path, dst_label_path)

    for sTs in tqdm(test_samples, desc="Testing cases"):
        sTs = sTs[: -len(".nii.gz")]
        sTs_valid = remove_punctuation_and_spaces(sTs)

        # Loading relevant modalities and the ground truth
        src_image_file_path1 = join(images_dir, sTs + "_0000.nii.gz")
        src_image_file_path2 = join(images_dir, sTs + "_0001.nii.gz")
        src_label_path = join(labels_dir, sTs + ".nii.gz")

        dst_image_file_path1 = f"{target_imagesTs}/{task_prefix}_{sTs_valid}_000.nii.gz"
        dst_image_file_path2 = f"{target_imagesTs}/{task_prefix}_{sTs_valid}_001.nii.gz"
        dst_label_path = f"{target_labelsTs}/{task_prefix}_{sTs_valid}.nii.gz"

        shutil.copy2(src_image_file_path1, dst_image_file_path1)
        shutil.copy2(src_image_file_path2, dst_image_file_path2)
        shutil.copy2(src_label_path, dst_label_path)

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        ("PET", "CT"),
        labels={0: "background", 1: "tumour"},
        dataset_name=task_name,
        license="CC BY-NC 4.0 DEED",
        dataset_description="AutoPET3 Multicenter Multitracer Generalization",
        dataset_reference="https://autopet-iii.grand-challenge.org/",
    )
