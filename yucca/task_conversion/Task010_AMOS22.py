from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.task_conversion.utils import generate_dataset_json
import shutil
from yucca.paths import yucca_raw_data


def convert(path: str, subdir: str = "amos22"):
    # INPUT DATA

    path = f"{path}/{subdir}"
    file_suffix = ".nii.gz"

    # Train/Test Splits
    labels_dir_tr = join(path, "labelsTr")
    images_dir_tr = join(path, "imagesTr")
    training_samples = subfiles(labels_dir_tr, join=False, suffix=file_suffix)

    labels_dir_ts = join(path, "labelsVa")
    images_dir_ts = join(path, "imagesVa")
    test_samples = subfiles(labels_dir_ts, join=False, suffix=file_suffix)

    ###OUTPUT DATA
    # Target names
    task_name = "Task010_AMOS22"
    prefix = "AMOS22"

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

    ###Populate Target Directory###
    # This is likely also the place to apply any re-orientation, resampling and/or label correction.
    for sTr in training_samples:
        serial_number = sTr[: -len(file_suffix)]
        shutil.copy2(join(images_dir_tr, sTr), f"{target_imagesTr}/{prefix}_{serial_number}_000.nii.gz")
        shutil.copy2(join(labels_dir_tr, sTr), f"{target_labelsTr}/{prefix}_{serial_number}.nii.gz")

    for sTs in test_samples:
        serial_number = sTs[: -len(file_suffix)]
        shutil.copy2(join(images_dir_ts, sTs), f"{target_imagesTs}/{prefix}_{serial_number}_000.nii.gz")
        shutil.copy2(join(labels_dir_ts, sTs), f"{target_labelsTs}/{prefix}_{serial_number}.nii.gz")

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        ("CT",),
        labels={
            "0": "background",
            "1": "spleen",
            "2": "right kidney",
            "3": "left kidney",
            "4": "gall bladder",
            "5": "esophagus",
            "6": "liver",
            "7": "stomach",
            "8": "arota",
            "9": "postcava",
            "10": "pancreas",
            "11": "right adrenal gland",
            "12": "left adrenal gland",
            "13": "duodenum",
            "14": "bladder",
            "15": "prostate/uterus",
        },
        dataset_name=task_name,
        license="hands off!",
        dataset_description="AMOS22",
        dataset_reference="",
    )
