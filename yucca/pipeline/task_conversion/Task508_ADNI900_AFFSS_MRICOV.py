# Dataset containing 150 AD and 150 controls used to test Classification pipeline.
# In this version we ONLY use the MRI and the labels.
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
import gzip
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p as ensure_dir_exists, isfile
from yucca.paths import get_raw_data_path
from yucca.pipeline.task_conversion.utils import generate_dataset_json
from sklearn.model_selection import train_test_split
from yucca.functional.utils.nib_utils import get_nib_orientation, reorient_nib_image


def convert(path: str = "/home/zcr545/data/data/projects/PsyBrainPrediction", subdir: str = "ADNI"):
    path = join(path, subdir)

    # Train and Test images are in the same folder
    images_dir = join(path, "Images")

    # OUTPUT DATA
    # Target names
    task_name = "Task508_ADNI900_AFFSS_MRICOV"
    prefix = "ADNI900_AFFSS_MRICOV"

    # Target paths
    target_base = join(get_raw_data_path(), task_name)

    target_imagesTr = join(target_base, "imagesTr")
    target_covariatesTr = join(target_base, "covariatesTr")
    target_labelsTr = join(target_base, "labelsTr")

    target_imagesTs = join(target_base, "imagesTs")
    target_covariatesTs = join(target_base, "covariatesTs")
    target_labelsTs = join(target_base, "labelsTs")

    ensure_dir_exists(target_imagesTr)
    ensure_dir_exists(target_covariatesTr)
    ensure_dir_exists(target_labelsTs)
    ensure_dir_exists(target_imagesTs)
    ensure_dir_exists(target_covariatesTs)
    ensure_dir_exists(target_labelsTr)

    # collect labels
    AD_cases = pd.read_csv(join(path, "AD_group.csv"))
    CN_cases = pd.read_csv(join(path, "CN_group.csv"))
    MCI_cases = pd.read_csv(join(path, "MCI_group.csv"))

    # Check that no subjects appear twice in either group
    assert len(set(AD_cases.Subject)) == 350
    assert len(set(CN_cases.Subject)) == 350
    assert len(set(MCI_cases.Subject)) == 350

    # Check that no subjects appear in both groups
    assert len(np.intersect1d(AD_cases.Subject, CN_cases.Subject)) == 0
    assert len(np.intersect1d(AD_cases.Subject, MCI_cases.Subject)) == 0
    assert len(np.intersect1d(CN_cases.Subject, MCI_cases.Subject)) == 0

    train_AD, test_AD = train_test_split(AD_cases, random_state=418920)
    train_CN, test_CN = train_test_split(CN_cases, random_state=537289)
    train_MCI, test_MCI = train_test_split(MCI_cases, random_state=958013)

    all_train = pd.concat([train_AD, train_CN, train_MCI])
    all_test = pd.concat([test_AD, test_CN, test_MCI])

    # Populate the training folders
    for _, row in tqdm(all_train.iterrows(), total=len(all_train)):
        subject = row["Subject"]
        label = row["Group"]
        age = row["Age"]
        sex = row["Sex"]

        # Fix the labels
        age = float(age / 122)  # 122 = oldest human ever will have value 1.0

        if sex == "F":
            sex = 0
        elif sex == "M":
            sex = 1
        else:
            print("Found unexpected sex: '", sex, "'")

        if label == "CN":
            label = 0
        elif label == "AD":
            label = 1
        elif label == "MCI":
            label = 2
        else:
            print("Found unexpected labels: '", label, "'")

        image_path = join(path, "MNI/Data_Affine_SkullStripped", subject, "T1_mni_affine.nii.gz")

        image_save_path = f"{target_imagesTr}/{prefix}_{subject}_000.nii.gz"
        cov_save_path = f"{target_covariatesTr}/{prefix}_{subject}_COV.txt"
        label_save_path = f"{target_labelsTr}/{prefix}_{subject}.txt"

        np.savetxt(cov_save_path, np.array([age, sex]), fmt="%s")
        np.savetxt(label_save_path, np.array([label]), fmt="%s")
        if not isfile(image_save_path):
            shutil.copy2(image_path, image_save_path)

    # Populate the test folders
    for _, row in tqdm(all_test.iterrows(), total=len(all_test)):
        subject = row["Subject"]
        label = row["Group"]
        age = row["Age"]
        sex = row["Sex"]

        # Fix the labels
        age = float(age / 122)  # 122 = oldest human ever will have value 1.0

        if sex == "F":
            sex = 0
        elif sex == "M":
            sex = 1
        else:
            print("Found unexpected sex: '", sex, "'")

        if label == "CN":
            label = 0
        elif label == "AD":
            label = 1
        elif label == "MCI":
            label = 2
        else:
            print("Found unexpected labels: ", label)

        image_path = join(path, "MNI/Data_Affine_SkullStripped", subject, "T1_mni_affine.nii.gz")

        image_save_path = f"{target_imagesTs}/{prefix}_{subject}_000.nii.gz"
        cov_save_path = f"{target_covariatesTs}/{prefix}_{subject}_COV.txt"
        label_save_path = f"{target_labelsTs}/{prefix}_{subject}.txt"

        np.savetxt(cov_save_path, np.array([age, sex]), fmt="%f")
        np.savetxt(label_save_path, np.array([label]), fmt="%f")
        if not isfile(image_save_path):
            shutil.copy2(image_path, image_save_path)

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        ("T1",),
        labels={0: "CN", 1: "AD", 2: "MCI"},
        im_ext="nii.gz",
        dataset_name=task_name,
        license="CC-BY 4.0",
        dataset_description="ADNI",
        dataset_reference="",
    )


if __name__ == "__main__":
    convert()
