import shutil
import gzip
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p as ensure_dir_exists, subfiles
from yucca.pipeline.task_conversion.utils import generate_dataset_json
from yucca.paths import get_raw_data_path, get_source_path


def convert(path: str = get_source_path(), subdir: str = "LITS"):
    """INPUT DATA - Define input path and suffixes"""
    path = join(path, subdir)
    file_suffix = ".nii"

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task019_LITS"
    task_prefix = "LITS"

    """ Access the input data. If images are not split into train/test, and you wish to randomly
    split the data, uncomment and adapt the following lines to fit your local path. """

    training_batches = [join(path, "Training Batch 1"), join(path, "Training Batch 2")]

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

    for tr_batch in training_batches:
        images_dir_tr = labels_dir_tr = tr_batch
        for sTr in subfiles(images_dir_tr, join=False):
            if "volume" not in sTr:
                continue
            case_id = sTr.split("-")[-1][: -len(file_suffix)]
            src_image_file = open(join(images_dir_tr, "volume-" + case_id + file_suffix), "rb")
            src_label_file = open(join(labels_dir_tr, "segmentation-" + case_id + file_suffix), "rb")
            dst_image_file_path = f"{target_imagesTr}/{task_prefix}_{case_id}_000.nii.gz"
            dst_label_file_path = f"{target_labelsTr}/{task_prefix}_{case_id}.nii.gz"

            shutil.copyfileobj(src_image_file, gzip.open(dst_image_file_path, "wb"))
            shutil.copyfileobj(src_label_file, gzip.open(dst_label_file_path, "wb"))

    # for sTs in subfiles(images_dir_ts, suffix=file_suffix, join=False):
    #    case_id = sTs[: -len(file_suffix)]
    #    src_image_file_path = join(images_dir_ts, case_id + file_suffix)
    #    dst_image_file_path = f"{target_imagesTs}/{task_prefix}_{case_id}_000.nii.gz"
    #    shutil.copy2(src_image_file_path, dst_image_file_path)

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        modalities=("CT",),
        labels={
            0: "background",
            1: "liver",
            2: "tumor",
        },
        dataset_name=task_name,
        license="",
        dataset_description="Liver Tumor Segmentation Challenge from Miccai 2017",
        dataset_reference="https://competitions.codalab.org/competitions/17094",
    )


if __name__ == "__main__":
    convert()
