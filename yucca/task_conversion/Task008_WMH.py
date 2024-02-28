import shutil
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfolders
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data


def convert(path: str, subdir: str = "WMH"):
    """INPUT DATA - Define input path and suffixes"""
    path = join(path, subdir)

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task008_WMH"
    task_prefix = "WMH"

    datasets = ["Amsterdam", "Singapore", "Utrecht"]
    site = ""

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

    ###Populate Target Directory###
    for dataset in datasets:
        dataset_path = join(path, dataset)
        if dataset == "Amsterdam":
            tr_folder = "Train_GE3T"
        else:
            tr_folder = "Train"

        train_folder = join(dataset_path, tr_folder)
        test_folder = join(dataset_path, "Test")

        training_samples = subfolders(train_folder, join=False)
        test_samples = subfolders(test_folder, join=False)

        # First we sort the training data
        for sTr in training_samples:
            # Loading relevant modalities and the ground truth
            src_image_file_path1 = join(train_folder, sTr, "pre", "FLAIR.nii.gz")
            src_image_file_path2 = join(train_folder, sTr, "pre", "T1.nii.gz")
            src_label_path = join(train_folder, sTr, "pre", "wmh.nii.gz")
            dst_image_file_path1 = f"{target_imagesTr}/{task_prefix}_{sTr}_000.nii.gz"
            dst_image_file_path2 = f"{target_imagesTr}/{task_prefix}_{sTr}_001.nii.gz"
            dst_label_path = f"{target_labelsTr}/{task_prefix}_{sTr}.nii.gz"
            shutil.copy2(src_image_file_path1, dst_image_file_path1)
            shutil.copy2(src_image_file_path2, dst_image_file_path2)
            shutil.copy2(src_label_path, dst_label_path)

        # Now we sort the test data
        if dataset == "Amsterdam":
            for site in test_samples:
                samples = subfolders(join(test_folder, site), join=False)
                for sTs in samples:
                    src_image_file_path1 = join(test_folder, site, sTs, "pre", "FLAIR.nii.gz")
                    src_image_file_path2 = join(test_folder, site, sTs, "pre", "T1.nii.gz")
                    src_label_path = join(test_folder, site, sTs, "pre", "wmh.nii.gz")
                    dst_image_file_path1 = f"{target_imagesTs}/{task_prefix}_{sTs}_000.nii.gz"
                    dst_image_file_path2 = f"{target_imagesTs}/{task_prefix}_{sTs}_001.nii.gz"
                    dst_label_path = f"{target_labelsTs}/{task_prefix}_{sTs}.nii.gz"
                    shutil.copy2(src_image_file_path1, dst_image_file_path1)
                    shutil.copy2(src_image_file_path2, dst_image_file_path2)
                    shutil.copy2(src_label_path, dst_label_path)

        else:
            for sTs in test_samples:
                src_image_file_path1 = join(test_folder, sTs, "pre", "FLAIR.nii.gz")
                src_image_file_path2 = join(test_folder, sTs, "pre", "T1.nii.gz")
                src_label_path = join(test_folder, sTs, "pre", "wmh.nii.gz")
                dst_image_file_path1 = f"{target_imagesTs}/{task_prefix}_{sTs}_000.nii.gz"
                dst_image_file_path2 = f"{target_imagesTs}/{task_prefix}_{sTs}_001.nii.gz"
                dst_label_path = f"{target_labelsTs}/{task_prefix}_{sTs}.nii.gz"
                shutil.copy2(src_image_file_path1, dst_image_file_path1)
                shutil.copy2(src_image_file_path2, dst_image_file_path2)
                shutil.copy2(src_label_path, dst_label_path)

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        ("Flair", "T1"),
        labels={0: "background", 1: "WMH", 2: "Other Pathology"},
        dataset_name=task_name,
        license="CC BY-NC 4.0 DEED",
        dataset_description="White Matter Hyperintensity Segmentation Challenge. Flair images only!",
        dataset_reference="https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/AECRSD",
    )
