import shutil
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfolders
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data


def convert(path: str, subdir: str = "WMH"):
    """INPUT DATA - Define input path and suffixes"""
    path = join(path, subdir)

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task208_WMH"
    task_prefix = "WMH"

    datasets = ["Amsterdam", "Singapore", "Utrecht"]
    # site = ""

    # Target paths
    target_base = join(yucca_raw_data, task_name)

    target_imagesTr = join(target_base, "imagesTr")

    maybe_mkdir_p(target_imagesTr)

    ###Populate Target Directory###
    for dataset in datasets:
        dataset_path = join(path, dataset)
        if dataset == "Amsterdam":
            tr_folder = "Train_GE3T"
        else:
            tr_folder = "Train"

        train_folder = join(dataset_path, tr_folder)
        # test_folder = join(dataset_path, "Test")

        training_samples = subfolders(train_folder, join=False)
        # test_samples = subfolders(test_folder, join=False)

        # First we add the data from the training folders
        for sTr in training_samples:
            # Loading relevant modalities and the ground truth
            src_flair_file_path = join(train_folder, sTr, "pre", "FLAIR.nii.gz")
            src_t1_file_path = join(train_folder, sTr, "pre", "T1.nii.gz")
            dst_flair_file_path = f"{target_imagesTr}/{task_prefix}_flair_{sTr}_000.nii.gz"
            dst_t1_file_path = f"{target_imagesTr}/{task_prefix}_t1_{sTr}_000.nii.gz"
            shutil.copy2(src_flair_file_path, dst_flair_file_path)
            shutil.copy2(src_t1_file_path, dst_t1_file_path)

        # Then we add the data from the original testing folders
        # if dataset == "Amsterdam":
        #   for site in test_samples:
        #       samples = subfolders(join(test_folder, site), join=False)
        #       for sTr in samples:
        #           # Loading relevant modalities and the ground truth
        #           src_flair_file_path = join(test_folder, site, sTr, "pre", "FLAIR.nii.gz")
        #           src_t1_file_path = join(test_folder, site, sTr, "pre", "T1.nii.gz")
        #           dst_flair_file_path = f"{target_imagesTr}/{task_prefix}_flair_{sTr}_000.nii.gz"
        #           dst_t1_file_path = f"{target_imagesTr}/{task_prefix}_t1_{sTr}_000.nii.gz"
        #           shutil.copy2(src_flair_file_path, dst_flair_file_path)
        #           shutil.copy2(src_t1_file_path, dst_t1_file_path)
    #
    # else:
    #   for sTr in test_samples:
    #       # Loading relevant modalities and the ground truth
    #       src_flair_file_path = join(test_folder, sTr, "pre", "FLAIR.nii.gz")
    #       src_t1_file_path = join(test_folder, sTr, "pre", "T1.nii.gz")
    #       dst_flair_file_path = f"{target_imagesTr}/{task_prefix}_flair_{sTr}_000.nii.gz"
    #       dst_t1_file_path = f"{target_imagesTr}/{task_prefix}_t1_{sTr}_000.nii.gz"
    #       shutil.copy2(src_flair_file_path, dst_flair_file_path)
    #       shutil.copy2(src_t1_file_path, dst_t1_file_path)

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        imagesTs_dir=None,
        modalities=("Flair/T1",),
        labels={},
        dataset_name=task_name,
        license="CC BY-NC 4.0 DEED",
        dataset_description="White Matter Hyperintensity Segmentation Challenge. Preprocessed Flair and T1 images and NO labels!",
        dataset_reference="https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/AECRSD",
    )
