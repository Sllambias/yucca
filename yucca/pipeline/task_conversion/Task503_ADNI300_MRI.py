# Dataset containing 150 AD and 150 controls used to test Classification pipeline.
# In this version we ONLY use the MRI and the labels.


import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm
import gzip
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p as ensure_dir_exists, isfile
from yucca.paths import get_raw_data_path
from yucca.pipeline.task_conversion.utils import generate_dataset_json
from sklearn.model_selection import train_test_split
from yucca.functional.utils.nib_utils import get_nib_orientation, reorient_nib_image
import nibabel as nib


def convert(path: str = "/home/zcr545/data/data/projects/PsyBrainPrediction", subdir: str = "ADNI"):
    path = join(path, subdir)

    # Train and Test images are in the same folder
    images_dir = join(path, "Images")

    # OUTPUT DATA
    # Target names
    task_name = "Task503_ADNI300_MRI"
    prefix = "ADNI300_MRI"

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

    # collect labels
    AD_cases = pd.read_csv(join(path, "AD_group.csv"))
    CN_cases = pd.read_csv(join(path, "CN_group.csv"))

    # Check that no subjects appear twice in either group
    assert len(set(AD_cases.Subject)) == 150
    assert len(set(CN_cases.Subject)) == 150

    # Check that no subjects appear in both groups
    assert len(np.intersect1d(AD_cases.Subject, CN_cases.Subject)) == 0

    train_AD, test_AD = train_test_split(AD_cases, random_state=418920)
    train_CN, test_CN = train_test_split(CN_cases, random_state=537289)

    all_train = pd.concat([train_AD, train_CN])
    all_test = pd.concat([test_AD, test_CN])

    # Populate the training folders
    for _, row in tqdm(all_train.iterrows(), total=len(all_train)):
        subject = row["Subject"]
        label = row["Group"]

        if label == "CN":
            label = 0
        elif label == "AD":
            label = 1
        else:
            print("Found unexpected labels: ", label)

        image_path = join(path, "Data", subject, "T1.nii")
        image_file = nib.load(image_path)

        ort = get_nib_orientation(image_file)
        image_file = reorient_nib_image(image_file, original_orientation=ort, target_orientation="RAS")

        image_save_path = f"{target_imagesTr}/{prefix}_{subject}_000.nii.gz"
        label_save_path = f"{target_labelsTr}/{prefix}_{subject}.txt"

        np.savetxt(label_save_path, np.array([label]), fmt="%s")
        if not isfile(image_save_path):
            nib.save(image_file, image_save_path)

    # Populate the test folders
    for _, row in tqdm(all_test.iterrows(), total=len(all_test)):
        subject = row["Subject"]
        label = row["Group"]

        if label == "CN":
            label = 0
        elif label == "AD":
            label = 1
        else:
            print("Found unexpected labels: ", label)

        image_path = join(path, "Data", subject, "T1.nii")
        image_file = nib.load(image_path)

        ort = get_nib_orientation(image_file)
        image_file = reorient_nib_image(image_file, original_orientation=ort, target_orientation="RAS")

        image_save_path = f"{target_imagesTs}/{prefix}_{subject}_000.nii.gz"
        label_save_path = f"{target_labelsTs}/{prefix}_{subject}.txt"

        np.savetxt(label_save_path, np.array([label]), fmt="%s")
        if not isfile(image_save_path):
            nib.save(image_file, image_save_path)

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        ("T1",),
        labels={0: "CN", 1: "AD"},
        dataset_name=task_name,
        license="CC-BY 4.0",
        dataset_description="ADNI",
        dataset_reference="",
    )


if __name__ == "__main__":
    convert()
