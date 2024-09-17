# ONLY used to run tests
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p as ensure_dir_exists, subfiles
from yucca.pipeline.task_conversion.utils import generate_dataset_json
import shutil
from yucca.paths import get_raw_data_path
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def convert(path: str, subdir: str = "dataset_test0_classification"):
    # INPUT DATA
    path = join(path, subdir)
    label_suffix = ".txt"
    image_suffix = ".nii.gz"

    # Train/Test Splits
    labels_dir = join(path, "all_labels")
    images_dir = join(path, "all_images")

    training_samples, test_samples = train_test_split(
        subfiles(labels_dir, join=False, suffix=label_suffix), random_state=859032
    )

    # OUTPUT DATA
    # Target names
    task_name = "Task000_TEST_CLASSIFICATION"
    task_prefix = "TEST_CLASSIFICATION"

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

    for sTr in tqdm(training_samples, desc="Train"):
        sTr = sTr[: -len(label_suffix)]
        src_image_file_path = join(images_dir, sTr + image_suffix)
        src_label_path = join(labels_dir, sTr + label_suffix)

        dst_image_file_path = f"{target_imagesTr}/{task_prefix}_{sTr}_000.nii.gz"
        dst_label_path = f"{target_labelsTr}/{task_prefix}_{sTr}.txt"
        shutil.copy2(src_image_file_path, dst_image_file_path)
        shutil.copy2(src_label_path, dst_label_path)
    del sTr

    for sTs in tqdm(test_samples, desc="Test"):
        sTs = sTs[: -len(label_suffix)]

        # Loading relevant modalities and the ground truth
        src_image_file_path = join(images_dir, sTs + image_suffix)
        src_label_path = join(labels_dir, sTs + label_suffix)

        dst_image_file_path = f"{target_imagesTs}/{task_prefix}_{sTs}_000.nii.gz"
        dst_label_path = f"{target_labelsTs}/{task_prefix}_{sTs}.nii.gz"
        shutil.copy2(src_image_file_path, dst_image_file_path)
        shutil.copy2(src_label_path, dst_label_path)
    del sTs

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        ("TestModality",),
        labels={0: "background", 1: "FakeLabel1"},
        dataset_name=task_name,
        dataset_description="Test Dataset",
        dataset_reference="",
    )


# %%
