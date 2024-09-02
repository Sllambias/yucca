import numpy as np
import os
import shutil
import nibabel as nib
from yucca.paths import get_yucca_raw_data
from typing import Literal
from batchgenerators.utilities.file_and_folder_operations import save_json, subfiles, join, subdirs
from tqdm import tqdm
from pathlib import Path


def combine_images_from_tasks(tasks: list, target_base: str, run_type: Literal["supervised", "unsupervised"]):
    assert len(tasks) > 0, "list of tasks empty"
    for task in tqdm(tasks):
        folders = ["imagesTr", "imagesTs", "labelsTr", "labelsTs"] if run_type == "supervised" else ["imagesTr"]
        for folder in folders:
            source = os.path.join(get_yucca_raw_data(), task, folder)
            target = os.path.join(target_base, folder)
            print("Copying ", source, target)
            copy_files_from_to(source, target)


def copy_files_from_to(source_dir, target_dir):
    assert os.path.isdir(source_dir)
    os.makedirs(target_dir, exist_ok=True)

    for file in tqdm(os.listdir(source_dir)):
        shutil.copy2(os.path.join(source_dir, file), f"{target_dir}/{file}")


def get_identifiers_from_splitted_files(folder: str, ext, tasks: list):
    if len(tasks) > 0:
        uniques = np.unique(
            [i[: -len("_000." + ext)] for task in tasks for i in subfiles(join(folder, task, suffix=ext), join=False)]
        )
    else:
        uniques = np.unique([i[: -len("_000." + ext)] for i in subfiles(folder, suffix=ext, join=False)])
    return list(uniques)


def dirs_in_dir(dir: str):
    p = Path(dir)
    return [f.name for f in p.iterdir() if f.is_dir() and f.name[0] not in [".", "_"]]


def files_in_dir(dir: str):
    p = Path(dir)
    return [f.name for f in p.iterdir() if f.is_file() and f.name[0] not in [".", "_"]]


def should_use_volume(vol: nib.Nifti1Image):
    return not (np.any(np.array(vol.shape) < 15) or len(vol.shape) != 3 or np.array(vol.dataobj).min() < 0)


def remove_punctuation_and_spaces(data: str):
    data = data.replace(" ", "-").replace(",", "-").replace(".", "-")
    return data


def generate_dataset_json(
    output_file: str,
    imagesTr_dir: str,
    imagesTs_dir: str,
    modalities: dict,
    labels: dict,
    dataset_name: str,
    label_hierarchy: dict = {},
    regions_in_order=[],  # We do not want to do these as dicts, as its sensitive to order and they easily become sorted by mistake
    regions_labeled=[],
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
    first_file = files_in_dir(imagesTr_dir)[0]
    im_ext = os.path.split(first_file)[-1].split(os.extsep, 1)[-1]
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
    json_dict["regions_in_order"] = regions_in_order
    json_dict["regions_labeled"] = regions_labeled
    json_dict["tasks"] = tasks
    json_dict["numTraining"] = len(train_identifiers)
    json_dict["numTest"] = len(test_identifiers)
    json_dict["training"] = [{"image": name, "label": name if labels else None} for name in train_identifiers]
    json_dict["test"] = test_identifiers

    if not output_file.endswith("dataset.json"):
        print(
            "WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
            "Proceeding anyways..."
        )
    save_json(json_dict, os.path.join(output_file))


def maybe_get_task_from_task_id(task_id: str | int):
    task_id = str(task_id)
    tasks = subdirs(get_yucca_raw_data(), join=False)

    # Check if name is already complete
    if task_id in tasks:
        return task_id

    # If not, we try to recreate the name
    # We use the raw_data folder as reference
    for task in tasks:
        if task_id.lower() in task.lower():
            return task

    # If we can't find anything we just return the original, on the offchance that the task does not exist in Raw Data while existing in e.g. Preprocessed
    print(
        f"Couldn't find a task called: {task_id} in the raw data folder: {get_yucca_raw_data()}. If your task only exists in e.g. the Preprocessed folder things might still work."
    )
    return task_id
