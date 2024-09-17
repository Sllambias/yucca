import numpy as np
import os
import shutil
import nibabel as nib
from yucca.paths import get_models_path, get_preprocessed_data_path, get_raw_data_path
from typing import Literal
from batchgenerators.utilities.file_and_folder_operations import save_json, subfiles, join, subdirs
from tqdm import tqdm
from pathlib import Path


def combine_images_from_tasks(tasks: list, target_base: str, run_type: Literal["supervised", "unsupervised"]):
    assert len(tasks) > 0, "list of tasks empty"
    for task in tqdm(tasks):
        folders = ["imagesTr", "imagesTs", "labelsTr", "labelsTs"] if run_type == "supervised" else ["imagesTr"]
        for folder in folders:
            source = os.path.join(get_raw_data_path(), task, folder)
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
    regions: dict = {},
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
    :param regions: dict with str -> dict where inner dict contains specification of each region
                    The dict must have the following form:
                    ```
                    {
                        'REGION_NAME': {
                            'priority': 1, # LOWER IS MORE IMPORTANT
                            'labels': ['LABEL1', 'LABEL2']
                        }
                        ...
                    }
                    where
                        priority specifies how regions are collapsed during prediction
                        labels specify which labels to be included into a region.
                               each label must be a key in the labels dict above.
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

    labels = {str(i): labels[i] for i in labels.keys()} if labels is not None else None

    # Sanity check regions
    if regions is not None:
        assert labels is not None
        assert isinstance(regions, dict), "Regions must be specified using a dict mapping regions to labels"
        priorities = []
        for region, region_dict in regions.items():
            assert "priority" in region_dict.keys(), f"Region {region} must have a priority specifier"
            priorities.append(region_dict["priority"])
            assert "labels" in region_dict.keys(), f"Region {region} must have labels"
            assert isinstance(region_dict["labels"], list), f"Region {region} labels must be a list"
            for label in region_dict["labels"]:
                assert isinstance(label, str), f"label must be of type str not {type(label)}"
                assert (
                    label in labels.values()
                ), f"Labels must be strings, corresponding to the values in the label dict: {labels.values()}"
                assert len(region_dict.keys()) == 2, region_dict.keys()

        # check that the priorities in the dict contains all numbers between 1 and max(priorities):
        assert set(priorities) == set(
            range(1, len(regions) + 1)
        ), f"The regions dict must contain all priorities between 1 and len(regions), but instead got priorities: {priorities}"

    json_dict = {}
    json_dict["name"] = dataset_name
    json_dict["description"] = dataset_description
    json_dict["tensorImageSize"] = "4D"
    json_dict["reference"] = dataset_reference
    json_dict["licence"] = license
    json_dict["release"] = dataset_release
    json_dict["image_extension"] = im_ext
    json_dict["modality"] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict["labels"] = labels
    json_dict["label_hierarchy"] = label_hierarchy
    json_dict["regions"] = regions
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


def get_task_from_task_id(task_id: str | int, stage: str):
    assert stage in ["raw", "preprocessed", "models"], stage
    task_id = str(task_id)

    if stage == "raw":
        stage_path = get_raw_data_path()
    elif stage == "preprocessed":
        stage_path = get_preprocessed_data_path()
    elif stage == "models":
        stage_path = get_models_path()

    tasks = subdirs(stage_path, join=False)

    # Check if name is already complete
    if task_id in tasks:
        return task_id

    # If not, we try to recreate the name
    # We use the raw_data folder as reference
    for task in tasks:
        if task_id.lower() in task.lower():
            return task

    raise LookupError(f"Task {task_id} not found in {stage_path}.")
