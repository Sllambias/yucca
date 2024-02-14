"""
[INTERNAL]

Task_conversion file for the SONAI data, without labels and with 2D jpg images (cache).
"""

import shutil
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from yucca.paths import yucca_raw_data
from yucca.task_conversion.utils import generate_dataset_json


def convert(path: str, txt_file_prefix: str = "data"):
    """
    Converts the SONAI image chache to the Yucca format.

    We are abusing the subdir argument to specify a file with the label paths.

    Usage:
        python run_task_conversion.py -t Task901_SONAI_NoLabel -p /path/to/SONAI -d data

        there need to be data_train.txt and data_test.txt files in the specified directory. (data is the default prefix)
    """

    # INPUT DATA
    images_tr_txt = join(path, f"{txt_file_prefix}_train.txt")
    images_ts_txt = join(path, f"{txt_file_prefix}_test.txt")

    # Image paths are in a txt that we read line by line
    with open(images_tr_txt, "r") as f:
        image_paths_tr = f.read().splitlines()

    with open(images_ts_txt, "r") as f:
        image_paths_ts = f.read().splitlines()

    # OUTPUT DATA
    # Target names
    task_name = "Task901_SONAI_NoLabel"
    prefix = "SONAI"

    # Target paths
    target_base = join(yucca_raw_data, task_name)

    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesTs)

    # Populate Target Directory

    for image_path in tqdm(image_paths_tr, desc="Train"):
        target_file_name = image_path.split("/")[-1][0:-4]  # filename without extension
        target_file_name = target_file_name.replace(".", "_")  # replace dots with underscores
        shutil.copyfile(image_path, f"{target_imagesTr}/{prefix}_{target_file_name}_000.jpg")

    for image_path in tqdm(image_paths_ts, desc="Test"):
        target_file_name = image_path.split("/")[-1][0:-4]
        target_file_name = target_file_name.replace(".", "_")
        shutil.copyfile(image_path, f"{target_imagesTs}/{prefix}_{target_file_name}_000.jpg")

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        ("US2D",),
        labels={},
        dataset_name=task_name,
        license="hands off!",
        dataset_description="SONAI Dataset from image cache",
        dataset_reference="",
    )
