import shutil
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p as ensure_dir_exists, subfiles
from yucca.pipeline.task_conversion.utils import generate_dataset_json
from yucca.paths import get_raw_data_path, get_source_path


def convert(path: str = get_source_path(), subdir: str = "BTCV_Abdomen"):
    """INPUT DATA - Define input path and suffixes"""
    path = join(path, subdir)
    file_suffix = ".nii.gz"

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task017_BTCV"
    task_prefix = "BTCV"

    """ Access the input data. If images are not split into train/test, and you wish to randomly
    split the data, uncomment and adapt the following lines to fit your local path. """

    images_dir_tr = join(path, "RawData", "Training", "img")
    labels_dir_tr = join(path, "RawData", "Training", "label")
    images_dir_ts = join(path, "RawData", "Testing", "img")

    """ Then define target paths """
    target_base = join(get_raw_data_path(), task_name)

    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")

    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")

    ensure_dir_exists(target_imagesTr)
    ensure_dir_exists(target_labelsTs)
    ensure_dir_exists(target_imagesTs)
    ensure_dir_exists(target_labelsTr)

    """Populate Target Directory
    This is also the place to apply any re-orientation, resampling and/or label correction."""

    for sTr in subfiles(images_dir_tr, suffix=file_suffix, join=False):
        case_id = sTr[: -len(file_suffix)]
        src_image_file_path = join(images_dir_tr, case_id + file_suffix)
        src_label_file_path = join(labels_dir_tr, "label" + case_id[3:] + file_suffix)
        dst_image_file_path = f"{target_imagesTr}/{task_prefix}_{case_id}_000.nii.gz"
        dst_label_file_path = f"{target_labelsTr}/{task_prefix}_{case_id}.nii.gz"

        shutil.copy2(src_image_file_path, dst_image_file_path)
        shutil.copy2(src_label_file_path, dst_label_file_path)

    for sTs in subfiles(images_dir_ts, suffix=file_suffix, join=False):
        case_id = sTs[: -len(file_suffix)]
        src_image_file_path = join(images_dir_ts, case_id + file_suffix)
        dst_image_file_path = f"{target_imagesTs}/{task_prefix}_{case_id}_000.nii.gz"

        shutil.copy2(src_image_file_path, dst_image_file_path)

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        modalities=("cine-MRI",),
        labels={
            0: "background",
            1: "spleen",
            2: "right kidney",
            3: "left kidney",
            4: "gallbladder",
            5: "esophagus",
            6: "liver",
            7: "stomach",
            8: "aorta",
            9: "inferior vena cava",
            10: "portal vein and splenic vein",
            11: "pancreas",
            12: "right adrenal gland",
            13: "left adrenal gland",
        },
        dataset_name=task_name,
        license="",
        dataset_description="",
        dataset_reference="",
    )


if __name__ == "__main__":
    convert()
