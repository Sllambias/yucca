import numpy as np
import os
import shutil
import gzip
from yucca.paths import yucca_raw_data
from typing import Tuple, Union
from batchgenerators.utilities.file_and_folder_operations import save_json, subfiles, join
from tqdm import tqdm


def combine_imagesTr_from_tasks(tasks: Union[list, tuple], target_dir):
    assert len(tasks) > 0, "list of tasks empty"
    for task in tasks:
        source_dir = os.path.join(yucca_raw_data, task, "imagesTr")
        assert os.path.isdir(source_dir)
        for sTr in tqdm(os.listdir(source_dir), task):
            shutil.copy2(os.path.join(source_dir, sTr), f"{target_dir}/{sTr}")


def get_identifiers_from_splitted_files(folder: str, ext, tasks: list):
    if len(tasks) > 0:
        uniques = np.unique([i[: -len("_000." + ext)] for task in tasks for i in subfiles(join(folder, task), join=False)])
    else:
        uniques = np.unique([i[: -len("_000." + ext)] for i in subfiles(folder, join=False)])
    return uniques


def generate_dataset_json(
    output_file: str,
    imagesTr_dir: str,
    imagesTs_dir: str,
    modalities: dict,
    labels: dict,
    dataset_name: str,
    label_hierarchy: dict = {},
    tasks: list = [],
    license: str = "hands off!",
    dataset_description: str = "",
    dataset_reference="",
    dataset_release="0.0",
):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: dict of modality names and their corresponding values. must be in the same order as the images (first entry
    corresponds to _000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'left hippocampus', 2: 'right hippocampus'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    im_ext = os.path.split(subfiles(imagesTr_dir)[0])[-1].split(os.extsep, 1)[-1]
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir, im_ext, tasks)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir, im_ext, tasks)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict["name"] = dataset_name
    json_dict["description"] = dataset_description
    json_dict["tensorImageSize"] = "4D"
    json_dict["reference"] = dataset_reference
    json_dict["licence"] = license
    json_dict["release"] = dataset_release
    json_dict["image_extension"] = im_ext
    json_dict["modality"] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict["labels"] = {str(i): labels[i] for i in labels.keys()} if labels is not None else None
    json_dict["label_hierarchy"] = label_hierarchy
    json_dict["tasks"] = tasks
    json_dict["numTraining"] = len(train_identifiers)
    json_dict["numTest"] = len(test_identifiers)
    json_dict["training"] = [{"image": f"./imagesTr/{i}.{im_ext}", "label": f"./labelsTr/{i}"} for i in train_identifiers]
    json_dict["test"] = [f"./imagesTs/{i}.{im_ext}" for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print(
            "WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
            "Proceeding anyways..."
        )
    save_json(json_dict, os.path.join(output_file))
