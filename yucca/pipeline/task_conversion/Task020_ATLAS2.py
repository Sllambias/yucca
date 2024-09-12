import shutil
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p as ensure_dir_exists, subfiles, subdirs
from yucca.pipeline.task_conversion.utils import generate_dataset_json
from yucca.paths import get_raw_data_path, get_source_path


def convert(path: str = get_source_path(), subdir: str = "ATLAS_2"):
    """INPUT DATA - Define input path and suffixes"""
    path = join(path, subdir)
    file_suffix = ".nii.gz"

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task020_ATLAS2"
    task_prefix = "ATLAS2"

    """ Access the input data. If images are not split into train/test, and you wish to randomly
    split the data, uncomment and adapt the following lines to fit your local path. """

    images_dir_tr = labels_dir_tr = join(path, "Training")
    # images_dir_ts = labels_dir_ts = join(path, "Testing")

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

    for dataset in subdirs(images_dir_tr, join=False):
        for sTr in subdirs(join(images_dir_tr, dataset), join=False):
            sessions = subdirs(join(images_dir_tr, dataset, sTr), join=False)
            assert len(sessions) == 1 and sessions[0] == "ses-1", f"unexpected # sessions found: {sessions}"
            image_types = subdirs(join(images_dir_tr, dataset, sTr, sessions[0]), join=False)
            assert len(image_types) == 1 and image_types[0] == "anat", f"unexpected type of scan. Found {image_types}"

            image_and_label = subfiles(join(images_dir_tr, dataset, sTr, sessions[0], image_types[0]), join=False)
            image = [i for i in image_and_label if "label" not in i][0]
            label = [i for i in image_and_label if "label" in i][0]
            case_id = image[: -len(file_suffix)]

            src_image_file_path = join(images_dir_tr, dataset, sTr, sessions[0], image_types[0], image)
            src_label_file_path = join(labels_dir_tr, dataset, sTr, sessions[0], image_types[0], label)

            label = nib.load(src_label_file_path)
            labelarr = label.get_fdata()
            labelarr[labelarr > 0] = 1
            label = nib.Nifti1Image(labelarr, label.affine, label.header)

            dst_image_file_path = f"{target_imagesTr}/{task_prefix}_{case_id}_000.nii.gz"
            dst_label_file_path = f"{target_labelsTr}/{task_prefix}_{case_id}.nii.gz"

            shutil.copy2(src_image_file_path, dst_image_file_path)
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
        modalities=("T1",),
        labels={
            0: "background",
            1: "lesion",
        },
        dataset_name=task_name,
        license="",
        dataset_description="ATLAS2 from MICCAI 2022",
        dataset_reference="https://atlas.grand-challenge.org/ATLAS/",
    )


if __name__ == "__main__":
    convert()
