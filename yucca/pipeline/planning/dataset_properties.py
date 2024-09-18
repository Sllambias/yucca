import logging
import nibabel as nib
import numpy as np
from typing import Dict, List
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, load_json, save_pickle, isfile
from yucca.functional.utils.nib_utils import get_nib_spacing
from yucca.functional.utils.type_conversions import nifti_or_np_to_np
from yucca.functional.utils.loading import read_file_to_nifti_or_np
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from functools import partial


def create_dataset_properties(data_dir, save_dir, suffix=".nii.gz", num_workers=12):
    """ "
    This will create a preprocessing-agnostic .pkl file containing basic dataset properties.
    We create this as a stand-alone file as the properties will be used by all preprocessors and should thus not
    be computed each time, as that will be very time consuming for memory intensive for larger datasets
    """
    dataset_json = load_json(join(data_dir, "dataset.json"))

    image_extension = dataset_json.get("image_extension") or suffix[1:]
    regions = dataset_json.get("regions") or None
    labels = dataset_json.get("labels") or None

    properties = {
        "image_extension": image_extension,
        "classes": list(dataset_json["labels"].keys()),
        "tasks": {task: [] for task in dataset_json["tasks"]},
        "label_hierarchy": dataset_json["label_hierarchy"],
        "modalities": dataset_json["modality"],
        "labels": labels,
        "regions": regions,
    }

    if len(dataset_json["tasks"]) > 0:
        assert not dataset_json["label_hierarchy"], "Multi Task implementation currently doesn't support Label Hierarchies"
        properties["classes"] = [list(dataset_json["labels"][task].keys()) for task in dataset_json["tasks"]]

    intensities = []
    sizes = []
    spacings = []
    background_pixel_values = []
    images_dir = join(data_dir, "imagesTr")
    labels_dir = join(data_dir, "labelsTr")
    modalities = properties["modalities"].items()
    mod_ids = list(list(zip(*modalities))[0])
    assert sorted(mod_ids) == mod_ids

    # Modalities are processed from smallest id to largest
    for mod_id, mod_name in modalities:
        mod_id = int(mod_id)
        suffix = f"_{mod_id:03}.{image_extension}"
        subjects = []
        images = subfiles(images_dir, suffix=suffix, join=False)

        for image in images:
            image_path = join(images_dir, image)
            # Remove modality encoding
            id = image[: -len(suffix)]
            expected_label_path = join(labels_dir, f"{id}.{image_extension}")
            if isfile(expected_label_path):
                subjects.append([image_path, expected_label_path])
            else:
                subjects.append([image_path, None])

        assert subjects, (
            f"no subjects found in {images_dir}. Ensure samples are "
            "suffixed with the modality encoding (such as '_000' for the first/only modality)"
            f"Looked for a files with ending {suffix}"
        )

        if "CT" in mod_name.upper():
            # -1000 == Air in HU (see https://en.wikipedia.org/wiki/Hounsfield_scale)
            # In practice it can be -1024, -1023 or in some cases -2048
            print("Using Hounsfield Units. Changing background pixel value from 0 to -1000")
            background_pixel_value = -1000
        else:
            background_pixel_value = 0
        background_pixel_values.append(background_pixel_value)

        chunksize = 50 if len(subjects) > 1000 else 1

        # `process_map` is a `tqdm` wrapper around `multiprocessing.Pool.map`
        map_result = process_map(
            partial(process, background_pixel_value=background_pixel_value),
            subjects,
            max_workers=num_workers,
            chunksize=chunksize,
            desc="Map",
        )  # returns list of dictionaries
        metadata = reduce(map_result)  # returns dictionary of metadata

        intensities.append(
            {
                "mean": float(np.mean(metadata["means"])),
                "min": float(np.min(metadata["mins"])),
                "max": float(np.max(metadata["maxs"])),
                "maxmin": float(np.max(metadata["mins"])),
                "std": float(np.mean(metadata["stds"])),
                "percentile_99_5": float(np.percentile(metadata["percentile_99_5"], 99.5)),
                "percentile_00_5": float(np.percentile(metadata["percentile_00_5"], 00.5)),
            }
        )

        # cross-modality metadata
        sizes = sizes + metadata["sizes"]
        spacings = spacings + metadata["spacings"]

    dims = [len(size) for size in sizes]
    assert all([dim == dims[0] for dim in dims]), f"not all volumes have the same number of dimensions. Sizes were: {dims}"

    properties["background_pixel_values"] = background_pixel_values
    properties["data_dimensions"] = int(len(sizes[0]))
    properties["original_sizes"] = sizes
    properties["original_spacings"] = spacings
    properties["intensities"] = intensities
    properties["original_max_size"] = np.max(sizes, 0).tolist()
    properties["original_min_size"] = np.min(sizes, 0).tolist()
    properties["original_median_size"] = np.median(sizes, 0).tolist()
    properties["original_max_spacing"] = np.max(spacings, 0).tolist()
    properties["original_min_spacing"] = np.min(spacings, 0).tolist()
    properties["original_median_spacing"] = np.median(spacings, 0).tolist()

    save_pickle(properties, join(save_dir, "dataset_properties.pkl"))


def reduce(results: List[Dict]):
    means = []
    mins = []
    maxs = []
    stds = []
    spacings = []
    sizes = []
    percentile_00_5 = []
    percentile_99_5 = []

    for res in tqdm(results, desc="Reduce"):
        means.append(res["mean"])
        mins.append(res["min"])
        maxs.append(res["max"])
        stds.append(res["std"])
        spacings.append(res["spacing"])
        sizes.append(res["size"])
        percentile_00_5.append(res["percentile_00_5"])
        percentile_99_5.append(res["percentile_99_5"])

    return {
        "means": means,
        "mins": mins,
        "maxs": maxs,
        "stds": stds,
        "spacings": spacings,
        "sizes": sizes,
        "percentile_00_5": percentile_00_5,
        "percentile_99_5": percentile_99_5,
    }


def process(subject: str, background_pixel_value: int = 0):
    image, label = subject
    try:
        image = read_file_to_nifti_or_np(image)
        dim = len(image.shape)

        if dim > 3:
            logging.warn(
                f"A volume has more than three dimensions. This is most often a mistake." f"Dims: {dim}, Vol: {image}"
            )

        size = image.shape

        if isinstance(image, nib.Nifti1Image):
            spacing = get_nib_spacing(image).tolist()
        else:
            spacing = [1.0] * dim

        image = nifti_or_np_to_np(image)
        if label is not None:
            label = read_file_to_nifti_or_np(label)
            label = nifti_or_np_to_np(label)
            mask = label > 0
            image_msk = image[mask]
        else:
            image_msk = image[image > background_pixel_value]

        mean = np.mean(image_msk)
        std = np.std(image_msk)

        mn = np.min(image_msk)
        mx = np.max(image_msk)
        percentile_00_5, percentile_99_5 = np.percentile(image_msk, np.array((0.5, 99.5)))

    except Exception as err:
        logging.warn(
            f"Could not read `{subject}`, got error `{err}`." "Suppressing to finalize, but you might need to act accordingly."
        )

    return {
        "size": size,
        "spacing": spacing,
        "min": mn,
        "max": mx,
        "mean": mean,
        "std": std,
        "percentile_00_5": percentile_00_5,
        "percentile_99_5": percentile_99_5,
    }
