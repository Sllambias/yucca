from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subdirs
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data
import shutil


def convert(path: str, subdir: str = "MSSEG1_2016"):
    # Target names
    task_name = "Task011_MSSEG1"
    task_prefix = "MSSEG1"

    ###OUTPUT DATA
    # Target paths
    target_base = join(yucca_raw_data, task_name)

    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")

    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    # MSSEG 2016
    # INPUT DATA
    # Input path and names
    base_in = join(path, subdir)
    file_suffix = ".nii.gz"

    # Train/Test Splits
    # We only use the train folder, as the test folder does not contain segmentations
    # to obtain those, submission to the challenge is required (it's from 2008, forget it)
    train_folder = join(base_in, "training")
    test_folder = join(base_in, "Testing")

    ###Populate Target Directory###
    for center in subdirs(train_folder, join=False):
        training_samples = subdirs(join(train_folder, center), join=False)
        for sTr in training_samples:
            src_image_file_path1 = join(train_folder, center, sTr, "Preprocessed_Data", "DP_preprocessed" + file_suffix)
            src_image_file_path2 = join(train_folder, center, sTr, "Preprocessed_Data", "FLAIR_preprocessed" + file_suffix)
            src_image_file_path3 = join(train_folder, center, sTr, "Preprocessed_Data", "GADO_preprocessed" + file_suffix)
            src_image_file_path4 = join(train_folder, center, sTr, "Preprocessed_Data", "T1_preprocessed" + file_suffix)
            src_image_file_path5 = join(train_folder, center, sTr, "Preprocessed_Data", "T2_preprocessed" + file_suffix)
            dst_image_file_path1 = f"{target_imagesTr}/{task_prefix}_{center}_{sTr}_000.nii.gz"
            dst_image_file_path2 = f"{target_imagesTr}/{task_prefix}_{center}_{sTr}_001.nii.gz"
            dst_image_file_path3 = f"{target_imagesTr}/{task_prefix}_{center}_{sTr}_002.nii.gz"
            dst_image_file_path4 = f"{target_imagesTr}/{task_prefix}_{center}_{sTr}_003.nii.gz"
            dst_image_file_path5 = f"{target_imagesTr}/{task_prefix}_{center}_{sTr}_004.nii.gz"

            src_label_path = join(train_folder, center, sTr, "Masks", "Consensus" + file_suffix)
            dst_label_path = f"{target_labelsTr}/{task_prefix}_{center}_{sTr}.nii.gz"

            shutil.copy2(src_image_file_path1, dst_image_file_path1)
            shutil.copy2(src_image_file_path2, dst_image_file_path2)
            shutil.copy2(src_image_file_path3, dst_image_file_path3)
            shutil.copy2(src_image_file_path4, dst_image_file_path4)
            shutil.copy2(src_image_file_path5, dst_image_file_path5)

            shutil.copy2(src_label_path, dst_label_path)

    for center in subdirs(test_folder, join=False):
        test_samples = subdirs(join(test_folder, center), join=False)
        for sTs in test_samples:
            src_image_file_path1 = join(test_folder, center, sTs, "Preprocessed_Data", "DP_preprocessed" + file_suffix)
            src_image_file_path2 = join(test_folder, center, sTs, "Preprocessed_Data", "FLAIR_preprocessed" + file_suffix)
            src_image_file_path3 = join(test_folder, center, sTs, "Preprocessed_Data", "GADO_preprocessed" + file_suffix)
            src_image_file_path4 = join(test_folder, center, sTs, "Preprocessed_Data", "T1_preprocessed" + file_suffix)
            src_image_file_path5 = join(test_folder, center, sTs, "Preprocessed_Data", "T2_preprocessed" + file_suffix)
            dst_image_file_path1 = f"{target_imagesTs}/{task_prefix}_{center}_{sTs}_000.nii.gz"
            dst_image_file_path2 = f"{target_imagesTs}/{task_prefix}_{center}_{sTs}_001.nii.gz"
            dst_image_file_path3 = f"{target_imagesTs}/{task_prefix}_{center}_{sTs}_002.nii.gz"
            dst_image_file_path4 = f"{target_imagesTs}/{task_prefix}_{center}_{sTs}_003.nii.gz"
            dst_image_file_path5 = f"{target_imagesTs}/{task_prefix}_{center}_{sTs}_004.nii.gz"

            src_label_path = join(test_folder, center, sTs, "Masks", "Consensus" + file_suffix)
            dst_label_path = f"{target_labelsTs}/{task_prefix}_{center}_{sTs}.nii.gz"

            shutil.copy2(src_image_file_path1, dst_image_file_path1)
            shutil.copy2(src_image_file_path2, dst_image_file_path2)
            shutil.copy2(src_image_file_path3, dst_image_file_path3)
            shutil.copy2(src_image_file_path4, dst_image_file_path4)
            shutil.copy2(src_image_file_path5, dst_image_file_path5)

            shutil.copy2(src_label_path, dst_label_path)

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        ("DP", "FLAIR", "GADO", "T1", "T2"),
        labels={0: "background", 1: "MS Lesion"},
        dataset_name=task_name,
        license="hands off!",
        dataset_description="MSSEG1",
        dataset_reference="https://www.nitrc.org/projects/msseg, https://arxiv.org/abs/2206.06694",
    )


if __name__ == "__main__":
    convert()
