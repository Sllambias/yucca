from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p as ensure_dir_exists, subfiles
from yucca.pipeline.task_conversion.utils import generate_dataset_json
import shutil
from yucca.paths import get_raw_data_path


def convert(path: str, subdir: str = "amos22"):
    # INPUT DATA

    path = f"{path}/{subdir}"
    file_suffix = ".nii.gz"

    # Train/Test Splits
    labels_dir_tr = join(path, "labelsTr")
    images_dir_tr = join(path, "imagesTr")
    training_samples = subfiles(labels_dir_tr, join=False, suffix=file_suffix)

    labels_dir_va = join(path, "labelsVa")
    images_dir_va = join(path, "imagesVa")
    validation_samples = subfiles(labels_dir_va, join=False, suffix=file_suffix)

    images_dir_ts = join(path, "imagesTs")
    test_samples = subfiles(images_dir_ts, join=False, suffix=file_suffix)

    ###OUTPUT DATA
    # Target names
    task_name = "Task010_AMOS22TEST"
    prefix = "AMOS22TEST"

    # Target paths
    target_base = join(get_raw_data_path(), task_name)

    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")

    target_imagesTs = join(target_base, "imagesTs")

    ensure_dir_exists(target_imagesTr)
    ensure_dir_exists(target_labelsTr)
    ensure_dir_exists(target_imagesTs)

    # This is likely also the place to apply any re-orientation, resampling and/or label correction.
    for sTr in training_samples:
        serial_number = sTr[: -len(file_suffix)]
        shutil.copy2(join(images_dir_tr, sTr), f"{target_imagesTr}/{prefix}_{serial_number}_000.nii.gz")
        shutil.copy2(join(labels_dir_tr, sTr), f"{target_labelsTr}/{prefix}_{serial_number}.nii.gz")

    # This is likely also the place to apply any re-orientation, resampling and/or label correction.
    for sVa in validation_samples:
        serial_number = sVa[: -len(file_suffix)]
        shutil.copy2(join(images_dir_va, sVa), f"{target_imagesTr}/{prefix}_{serial_number}_000.nii.gz")
        shutil.copy2(join(labels_dir_va, sVa), f"{target_labelsTr}/{prefix}_{serial_number}.nii.gz")

    for sTs in test_samples:
        serial_number = sTs[: -len(file_suffix)]
        shutil.copy2(join(images_dir_ts, sTs), f"{target_imagesTs}/{prefix}_{serial_number}_000.nii.gz")

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
