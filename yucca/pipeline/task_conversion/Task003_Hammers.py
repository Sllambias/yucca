from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.task_conversion.utils import generate_dataset_json
import shutil
import gzip
from yucca.paths import yucca_raw_data
from tqdm import tqdm


def convert(path: str, subdir: str = "Hammers"):
    # INPUT DATA
    path = f"{path}/{subdir}"
    file_suffix = ".nii"

    # Train/Test Splits
    labels_dir = join(path, "Labels")
    images_dir = join(path, "Images")

    # S01 to S30 are used as training samples
    # S31 to S40 are used as test samples
    all_samples = subfiles(labels_dir, join=False, suffix=file_suffix)

    assert len(all_samples) > 0

    tr_ids = ["a" + f"{id:02}" for id in range(1, 21)]
    training_samples = [sample for sample in all_samples if sample[: -len(file_suffix)] in tr_ids]
    ts_ids = ["a" + f"{id:02}" for id in range(21, 31)]
    test_samples = [sample for sample in all_samples if sample[: -len(file_suffix)] in ts_ids]

    assert len(training_samples) > 0
    assert len(test_samples) > 0

    # OUTPUT DATA
    # Target names
    task_name = "Task003_Hammers"
    prefix = "Hammers"

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

    # Populate Target Directory
    # This is likely also the place to apply any re-orientation, resampling and/or label correction.
    for sTr in tqdm(training_samples):
        serial_number = sTr[:-4]
        image_file = open(join(images_dir, sTr), "rb")
        label = open(join(labels_dir, sTr), "rb")
        shutil.copyfileobj(image_file, gzip.open(f"{target_imagesTr}/{prefix}_{serial_number}_000.nii.gz", "wb"))
        shutil.copyfileobj(label, gzip.open(f"{target_labelsTr}/{prefix}_{serial_number}.nii.gz", "wb"))

    for sTs in tqdm(test_samples):
        serial_number = sTs[:-4]
        image_file = open(join(images_dir, sTs), "rb")
        label = open(join(labels_dir, sTs), "rb")
        shutil.copyfileobj(image_file, gzip.open(f"{target_imagesTs}/{prefix}_{serial_number}_000.nii.gz", "wb"))
        shutil.copyfileobj(label, gzip.open(f"{target_labelsTs}/{prefix}_{serial_number}.nii.gz", "wb"))

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        ("T1",),
        labels={0: "background", 1: "Left Hippocampus", 2: "Right Hippocampus"},
        dataset_name=task_name,
        license="hands off!",
        dataset_description="Hammers Dataset",
        dataset_reference="",
    )
