"""
Task_conversion file for the Fetal Planes DB (Barcelona) dataset (https://doi.org/10.5281/zenodo.3904280).

This is an example of a classification dataset with 2D PNG images and image-level labels.
"""

import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subdirs, subfiles
from yucca.paths import yucca_raw_data
from yucca.task_conversion.utils import generate_dataset_json


def convert(path: str, subdir: str = "tiny-imagenet-200"):
    # INPUT DATA

    path = f"{path}/{subdir}"
    # file_suffix = ".png"

    # Train and Val (We use Val as Test) images are in the same folder
    train_images_dir = join(path, "train")
    test_images_dir = join(path, "val", "images")
    # OUTPUT DATA
    # Target names
    task_name = "Task502_tiny_imagenet_200"
    prefix = "tiny_imagenet_200"

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
    next_label = 1
    labeldict = {}

    val_annotations = pd.read_csv(
        join(path, "val", "val_annotations.txt"),
        sep="\t",
        header=None,
        usecols=[0, 1],
    )

    train_categories = subdirs(train_images_dir, join=False)
    for category in tqdm(train_categories, total=len(train_categories)):
        if labeldict.get(category) is None:
            labeldict[category] = next_label
            next_label += 1
        label = np.array([labeldict[category]], dtype=np.uint8)
        for image in subfiles(join(train_images_dir, category, "images"), join=False):
            serial_number = image[: -len(".JPEG")]
            image_file_path = join(train_images_dir, category, "images", image)
            shutil.copyfile(image_file_path, f"{target_imagesTr}/{prefix}_{serial_number}_000.png")
            np.savetxt(f"{target_labelsTr}/{prefix}_{serial_number}.txt", label, fmt="%d")

    test_images = subfiles(test_images_dir, join=False)
    for image in tqdm(test_images, total=len(test_images)):
        category = val_annotations[1].iloc[np.where(val_annotations[0] == image)].values[0]
        assert labeldict.get(category) is not None
        label = np.array([labeldict[category]], dtype=np.uint8)
        serial_number = image[: -len(".JPEG")]
        image_file_path = join(test_images_dir, image)
        shutil.copyfile(image_file_path, f"{target_imagesTs}/{prefix}_{serial_number}_000.png")
        np.savetxt(f"{target_labelsTs}/{prefix}_{serial_number}.txt", label, fmt="%d")

    # Now construct the pair of label-ids and real labels

    label_names = pd.read_csv(
        join(path, "words.txt"),
        sep="\t",
        header=None,
        usecols=[0, 1],
    )
    labeldict_with_words = {}
    for key, val in labeldict.items():
        word = label_names[1].iloc[np.where(label_names[0] == key)].values[0]
        labeldict_with_words[val] = key + " = " + word
    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        ("Natural Images",),
        labels=labeldict_with_words,
        dataset_name=task_name,
        license="CC-BY 4.0",
        dataset_description="Tiny ImageNet 200",
        dataset_reference="",
    )
