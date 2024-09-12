import numpy as np
from typing import Optional, List, Union
from batchgenerators.utilities.file_and_folder_operations import subfiles, load_pickle


def make_plans_file(
    allow_missing_modalities: bool,
    crop_to_nonzero: bool,
    classes: list[Union[int, str]],
    norm_op: str,
    modalities: List[str],
    plans_name: str,
    dataset_properties: Optional[dict] = {},
    keep_aspect_ratio_when_using_target_size: bool = True,
    preprocessor: Optional[str] = None,
    task_type: str = "segmentation",
    target_coordinate_system: str = "RAS",
    target_spacing: list = [1.0, 1.0, 1.0],
    target_size: Optional[list] = None,
    transpose_forward: list = [0, 1, 2],
    transpose_backward: list = [0, 1, 2],
    suggested_dimensionality: str = "3D",
):
    assert task_type in ["classification", "segmentation", "self-supervised"]
    plans = {}
    plans["target_coordinate_system"] = target_coordinate_system
    plans["preprocessor"] = preprocessor
    # When True all the background between the volumes and the edges of the image is removed
    plans["crop_to_nonzero"] = crop_to_nonzero

    # Change this to have different normalization schemes for all or some modalities
    plans["normalization_scheme"] = norm_op

    plans["num_classes"] = len(classes)
    plans["num_modalities"] = len(modalities)

    plans["transpose_forward"] = transpose_forward
    plans["transpose_backward"] = transpose_backward

    # Defaults to the median spacing of the training data.
    # Change the determine_spacing() function to use different spacings
    plans["keep_aspect_ratio_when_using_target_size"] = keep_aspect_ratio_when_using_target_size
    plans["target_size"] = target_size
    plans["target_spacing"] = target_spacing
    plans["task_type"] = task_type
    plans["dataset_properties"] = dataset_properties
    plans["plans_name"] = plans_name
    plans["suggested_dimensionality"] = suggested_dimensionality
    plans["allow_missing_modalities"] = allow_missing_modalities
    return plans


def add_stats_to_plans_post_preprocessing(plans, directory):
    pkl_files = subfiles(directory, suffix=".pkl")

    new_spacings = []
    new_sizes = []
    n_cc = []
    size_cc = []
    for pkl_file in pkl_files:
        pkl_file = load_pickle(pkl_file)
        new_spacings.append(pkl_file["new_spacing"])
        new_sizes.append(pkl_file["new_size"])
        n_cc.append(pkl_file["label_cc_n"])
        if np.mean(pkl_file["label_cc_sizes"]) > 0:
            size_cc.append(np.mean(pkl_file["label_cc_sizes"], dtype=int))

    mean_size = np.mean(new_sizes, 0, dtype=int).tolist()
    min_size = np.min(new_sizes, 0).tolist()
    max_size = np.max(new_sizes, 0).tolist()

    if len(size_cc) > 0:
        mean_cc = np.mean(size_cc, dtype=int).tolist()
        min_cc = np.min(size_cc).tolist()
        max_cc = np.max(size_cc).tolist()
        mean_n_cc = np.mean(n_cc, dtype=int).tolist()
        min_n_cc = np.min(n_cc, initial=-1).tolist()
        max_n_cc = np.max(n_cc, initial=-1).tolist()
    else:
        mean_cc = min_cc = max_cc = mean_n_cc = min_n_cc = max_n_cc = 0

    plans["new_sizes"] = new_sizes
    plans["new_spacings"] = new_spacings
    plans["new_mean_size"] = mean_size
    plans["new_min_size"] = min_size
    plans["new_max_size"] = max_size
    plans["mean_cc_size"] = mean_cc
    plans["max_cc_size"] = max_cc
    plans["min_cc_size"] = min_cc
    plans["mean_n_cc"] = mean_n_cc
    plans["max_n_cc"] = max_n_cc
    plans["min_n_cc"] = min_n_cc
    return plans
