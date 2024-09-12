import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p as ensure_dir_exists, subdirs
from yucca.pipeline.task_conversion.utils import generate_dataset_json
from yucca.paths import get_raw_data_path, get_source_path
from yucca.functional.utils.nib_utils import reorient_to_RAS


def convert(path: str = get_source_path(), subdir: str = "kits23"):
    """INPUT DATA - Define input path and suffixes"""
    path = join(path, subdir)
    file_suffix = ".nii.gz"

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task018_KITS23"
    task_prefix = "KITS23"

    """ Access the input data. If images are not split into train/test, and you wish to randomly
    split the data, uncomment and adapt the following lines to fit your local path. """

    images_dir_tr = labels_dir_tr = join(path, "dataset")

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

    for sTr in subdirs(images_dir_tr, join=False):
        src_image_file_path = join(images_dir_tr, sTr, "imaging" + file_suffix)
        src_label_file_path = join(labels_dir_tr, sTr, "segmentation" + file_suffix)

        dst_image_file_path = f"{target_imagesTr}/{task_prefix}_{sTr}_000.nii.gz"
        dst_label_file_path = f"{target_labelsTr}/{task_prefix}_{sTr}.nii.gz"

        image = nib.load(src_image_file_path)
        label = nib.load(src_label_file_path)

        image = reorient_to_RAS(image)
        label = reorient_to_RAS(label)

        nib.save(image, dst_image_file_path)
        nib.save(label, dst_label_file_path)

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
        labels={0: "background", 1: "kidney", 2: "tumor", 3: "cyst"},
        regions_in_order=[[1, 2, 3], [2, 3], [2]],
        regions_labeled=[1, 3, 2],
        dataset_name=task_name,
        license="",
        dataset_description="The 2023 Kidney and Kidney Tumor Segmentation Challenge",
        dataset_reference="https://kits-challenge.org/kits23",
    )


if __name__ == "__main__":
    convert()
