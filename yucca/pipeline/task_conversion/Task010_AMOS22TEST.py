from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.pipeline.task_conversion.utils import generate_dataset_json
import shutil
from yucca.paths import yucca_raw_data


def convert(path: str, subdir: str = "amos22"):
    # INPUT DATA

    path = f"{path}/{subdir}"
    file_suffix = ".nii.gz"

    images_dir_ts = join(path, "imagesTs")
    test_samples = subfiles(images_dir_ts, join=False, suffix=file_suffix)

    ###OUTPUT DATA
    # Target names
    task_name = "Task010_AMOS22TEST"
    prefix = "AMOS22TEST"

    # Target paths
    target_base = join(yucca_raw_data, task_name)

    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")

    maybe_mkdir_p(target_imagesTs)

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
