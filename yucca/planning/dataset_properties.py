from batchgenerators.utilities.file_and_folder_operations import subfiles, join, load_json, save_pickle
from yuccalib.utils.nib_utils import get_nib_spacing
from yuccalib.utils.type_conversions import nifti_or_np_to_np, read_file_to_nifti_or_np
import nibabel as nib
import numpy as np
import sys


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
    im_ext = dataset_json.get("image_extension") or "nii.gz"
    properties["image_extension"] = im_ext
    properties["classes"] = list(dataset_json["labels"].keys())
    properties["tasks"] = {task: [] for task in dataset_json["tasks"]}
    if len(dataset_json["tasks"]) > 0:
        assert not dataset_json["label_hierarchy"], "Multi Task implementation currently doesn't support Label Hierarchies"
        properties["classes"] = [list(dataset_json["labels"][task].keys()) for task in dataset_json["tasks"]]

    properties["label_hierarchy"] = dataset_json["label_hierarchy"]
    properties["modalities"] = dataset_json["modality"]

    total_scans = dataset_json["numTraining"] * len(properties["modalities"])
    flushpoints = np.linspace(0, total_scans - 1, 6, dtype=int)[1:]
    completed = 0
    sizes = []
    spacings = []
    intensity_results = []

    means = []
    min = np.inf
    max = -np.inf
    stds = []

    for mod_id, mod_name in properties["modalities"].items():
        mod_id = int(mod_id)
        intensity_results.append({})

        subjects = subfiles(join(data_dir, "imagesTr"), suffix=f"_{mod_id:03}.{im_ext}")
        if len(dataset_json["tasks"]) > 0:
            for task in dataset_json["tasks"]:
                for subject in subfiles(join(data_dir, "imagesTr", task), suffix=f"_{mod_id:03}.{im_ext}", join=False):
                    if subject.endswith(f"_{mod_id:03}.{im_ext}"):
                        properties["tasks"][task].append(subject[: -len(f"_{mod_id:03}.{im_ext}")])
                    subjects.append(join(data_dir, "imagesTr", task, subject))

        assert subjects, (
            f"no subjects found in {join(data_dir, 'imagesTr')}. Ensure samples are "
            "suffixed with the modality encoding (such as '_000' for the first/only modality)"
        )
        background_pixel_value = 0
        if "CT" in mod_name.upper():
            # -1000 == Air in HU (see https://en.wikipedia.org/wiki/Hounsfield_scale)
            # In practice it can be -1024, -1023 or in some cases -2048
            print("Using Hounsfield Units. Changing background pixel value from 0 to -1000")
            background_pixel_value = -1000
        for idx, subject in enumerate(subjects):
            if idx * (mod_id + 1) in flushpoints:
                completed += 20
                print(f"Property file creation progress: {completed}%")
                sys.stdout.flush()
            image = read_file_to_nifti_or_np(subject)
            sizes.append(image.shape)
            if isinstance(image, nib.Nifti1Image):
                spacings.append(get_nib_spacing(image).tolist())
            else:
                spacings.append([1.0, 1.0, 1.0])
            image = nifti_or_np_to_np(image)
            mask = image >= background_pixel_value

            means.append(np.mean(image[mask]))
            stds.append(np.std(image[mask]))
            min = np.min([min, np.min(image[mask])])
            max = np.max([max, np.max(image[mask])])

        intensity_results[mod_id]["mean"] = float(np.mean(means))
        intensity_results[mod_id]["min"] = float(min)
        intensity_results[mod_id]["max"] = float(max)
        intensity_results[mod_id]["std"] = float(np.mean(stds))

    assert all([len(size) == len(sizes[0]) for size in sizes]), "not all volumes have the same" " number of dimensions"

    properties["data_dimensions"] = int(len(sizes[0]))
    properties["original_sizes"] = sizes
    properties["original_spacings"] = spacings
    properties["intensities"] = intensity_results
    save_pickle(properties, join(save_dir, "dataset_properties.pkl"))
