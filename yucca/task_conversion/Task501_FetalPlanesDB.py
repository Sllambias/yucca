"""
Task_conversion file for the Fetal Planes DB (Barcelona) dataset (https://doi.org/10.5281/zenodo.3904280).

This is an example of a classification dataset with 2D PNG images and image-level labels.
"""

import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from yucca.paths import yucca_raw_data
from yucca.task_conversion.utils import generate_dataset_json


def convert(path: str, subdir: str = "FETAL_PLANES_DB"):
    # INPUT DATA

    path = f"{path}/{subdir}"
    # file_suffix = ".png"

    # Train and Test images are in the same folder
    images_dir = join(path, "Images")

    # OUTPUT DATA
    # Target names
    task_name = "Task501_FetalPlanesDB"
    prefix = "FetalPlanesDB"

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

    # collect labels
    data_df = pd.read_csv(join(path, "FETAL_PLANES_DB_data.csv"), delimiter=";")

    # Since there's trailing whitespaces in the columns of the dataset we strip names.
    data_df = data_df.rename(columns=lambda x: x.strip())
    labels = list(data_df["Plane"].unique())

    # Populate Target Directory
    # This is likely also the place to apply any re-orientation, resampling and/or label correction.
    for index, row in tqdm(data_df.iterrows(), total=len(data_df)):
        serial_number = index
        image_file_path = join(images_dir, f"{row['Image_name']}.png")
        label = np.array([int(labels.index(row["Plane"]))], dtype=np.int64)
        is_train_split = row["Train"] == 1
        if is_train_split:
            shutil.copyfile(image_file_path, f"{target_imagesTr}/{prefix}_{serial_number}_000.png")
            np.savetxt(f"{target_labelsTr}/{prefix}_{serial_number}.txt", label, fmt="%d")
        else:
            shutil.copyfile(image_file_path, f"{target_imagesTs}/{prefix}_{serial_number}_000.png")
            np.savetxt(f"{target_labelsTs}/{prefix}_{serial_number}.txt", label, fmt="%d")

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        ("US2D",),
        labels={i: l for i, l in enumerate(labels)},
        dataset_name=task_name,
        license="CC-BY 4.0",
        dataset_description="Fetal Planes DB (Barcelona)",
        dataset_reference="https://doi.org/10.5281/zenodo.3904280",
    )
