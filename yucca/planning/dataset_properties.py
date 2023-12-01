import warnings
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, load_json, save_pickle
from yuccalib.utils.nib_utils import get_nib_spacing
from yuccalib.utils.type_conversions import nifti_or_np_to_np, read_file_to_nifti_or_np
import nibabel as nib
import numpy as np
import json
from tqdm import tqdm


def create_properties_pkl(data_dir, save_dir, suffix=".nii.gz"):
    """ "
    This will create a preprocessing-agnostic .pkl file containing basic dataset properties.
    We create this as a stand-alone file as the properties will be used by all preprocessors and should thus not
    be computed each time, as that will be very time consuming for memory intensive for larger datasets
    """
    properties = {}

    dataset_json = load_json(join(data_dir, "dataset.json"))
    im_ext = dataset_json.get("image_extension") or "nii.gz"
    properties["image_extension"] = im_ext
    properties["classes"] = list(dataset_json["labels"].keys())
    properties["tasks"] = {task: [] for task in dataset_json["tasks"]}
    if len(dataset_json["tasks"]) > 0:
        assert not dataset_json["label_hierarchy"], "Multi Task implementation currently doesn't support Label Hierarchies"
        properties["classes"] = [list(dataset_json["labels"][task].keys()) for task in dataset_json["tasks"]]

    properties["label_hierarchy"] = dataset_json["label_hierarchy"]
    properties["modalities"] = dataset_json["modality"]

    sizes = []
    spacings = []
    intensity_results = []

    means_fg = []
    min_fg = np.inf
    max_fg = -np.inf
    stds_fg = []

    means = []
    min = np.inf
    max = -np.inf
    stds = []

    for mod_id, mod_name in properties["modalities"].items():
        mod_id = int(mod_id)
        intensity_results.append({})
        suffix = f"_{mod_id:03}.{im_ext}"
        images_dir = join(data_dir, "imagesTr")

        subjects = subfiles(images_dir, suffix=suffix)
        if len(dataset_json["tasks"]) > 0:
            for task in dataset_json["tasks"]:
                for subject in subfiles(join(data_dir, "imagesTr", task), suffix=suffix, join=False):
                    if subject.endswith(suffix):
                        properties["tasks"][task].append(subject[: -len(suffix)])
                    subjects.append(join(data_dir, "imagesTr", task, subject))

        assert subjects, (
            f"no subjects found in {images_dir}. Ensure samples are "
            "suffixed with the modality encoding (such as '_000' for the first/only modality)"
            f"Looked for a files with ending {suffix}"
        )
        background_pixel_value = 0

        if "CT" in mod_name.upper():
            # -1000 == Air in HU (see https://en.wikipedia.org/wiki/Hounsfield_scale)
            # In practice it can be -1024, -1023 or in some cases -2048
            print("Using Hounsfield Units. Changing background pixel value from 0 to -1000")
            background_pixel_value = -1000

        # Loop through all images in task
        for subject in tqdm(subjects):
            image = read_file_to_nifti_or_np(subject)
            sizes.append(image.shape)
            dim = len(image.shape)
            if dim > 3:
                warnings.warn(
                    f"A volume has more than three dimensions. This is most often a mistake." f"Dims: {dim}, Vol: {subject}"
                )

            if isinstance(image, nib.Nifti1Image):
                spacings.append(get_nib_spacing(image).tolist())
            else:
                spacings.append([1.0, 1.0, 1.0])
            image = nifti_or_np_to_np(image)

            mask = image > background_pixel_value

            means.append(np.mean(image))
            means_fg.append(np.mean(image[mask]))
            stds.append(np.std(image))
            stds_fg.append(np.std(image[mask]))

            maxmin = np.max([min, np.min(image)])
            min = np.min([min, np.min(image)])
            max = np.max([max, np.max(image)])

            maxmin_fg = np.max([min_fg, np.min(image[mask])])
            min_fg = np.min([min_fg, np.min(image[mask])])
            max_fg = np.max([max_fg, np.max(image[mask])])

        intensity_results[mod_id]["fg"] = {
            "mean": float(np.mean(means_fg)),
            "min": float(min_fg),
            "max": float(max_fg),
            "maxmin": float(maxmin_fg),
            "std": float(np.mean(stds_fg)),
        }

        intensity_results[mod_id]["all"] = {
            "mean": float(np.mean(means)),
            "min": float(min),
            "max": float(max),
            "maxmin": float(maxmin),
            "std": float(np.mean(stds)),
        }

    dims = [len(size) for size in sizes]
    assert all([dim == dims[0] for dim in dims]), f"not all volumes have the same number of dimensions. Sizes were: {dims}"

    properties["data_dimensions"] = int(len(sizes[0]))
    properties["original_sizes"] = sizes
    properties["original_spacings"] = spacings
    properties["intensities"] = intensity_results
    save_pickle(properties, join(save_dir, "dataset_properties.pkl"))
    json.dump(intensity_results, open(join(save_dir, "intensity_results.json"), "w"))
