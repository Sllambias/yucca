import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p as ensure_dir_exists, subfiles, subdirs
from yucca.pipeline.task_conversion.utils import generate_dataset_json
from yucca.paths import get_raw_data_path, get_source_path


def convert(path: str = get_source_path(), subdir: str = "ACDC"):
    """INPUT DATA - Define input path and suffixes"""
    path = join(path, subdir)
    file_suffix = ".nii.gz"

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task016_ACDC"
    task_prefix = "ACDC"

    """ Access the input data. If images are not split into train/test, and you wish to randomly
    split the data, uncomment and adapt the following lines to fit your local path. """

    images_dir_tr = labels_dir_tr = join(path, "database", "training")
    images_dir_ts = labels_dir_ts = join(path, "database", "testing")

    train_samples = subdirs(images_dir_tr, join=False)
    test_samples = subdirs(images_dir_ts, join=False)

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

    for patient in train_samples:
        for sTr in subfiles(join(images_dir_tr, patient), join=False, suffix=file_suffix):
            if "_4d" in sTr or "_gt" in sTr:
                # skip these. We'll infer the label path from the remaining subjects
                continue
            case_id = sTr[: -len(file_suffix)]
            src_image_file_path = join(images_dir_tr, patient, case_id + file_suffix)
            src_label_file_path = join(labels_dir_tr, patient, case_id + "_gt" + file_suffix)
            dst_image_file_path = f"{target_imagesTr}/{task_prefix}_{case_id}_000.nii.gz"
            dst_label_file_path = f"{target_labelsTr}/{task_prefix}_{case_id}.nii.gz"

            image = nib.load(src_image_file_path)
            label = nib.load(src_label_file_path)

            image.set_sform(None)
            label.set_sform(None)

            nib.save(image, dst_image_file_path)
            nib.save(label, dst_label_file_path)

    del sTr

    for patient in test_samples:
        for sTs in subfiles(join(images_dir_ts, patient), join=False, suffix=file_suffix):
            if "_4d" in sTs or "_gt" in sTs:
                # skip these. We'll infer the label path from the remaining subjects
                continue
            case_id = sTs[: -len(file_suffix)]
            src_image_file_path = join(images_dir_ts, patient, case_id + file_suffix)
            src_label_file_path = join(labels_dir_ts, patient, case_id + "_gt" + file_suffix)
            dst_image_file_path = f"{target_imagesTs}/{task_prefix}_{case_id}_000.nii.gz"
            dst_label_file_path = f"{target_labelsTs}/{task_prefix}_{case_id}.nii.gz"

            image = nib.load(src_image_file_path)
            label = nib.load(src_label_file_path)

            image.set_sform(None)
            label.set_sform(None)

            nib.save(image, dst_image_file_path)
            nib.save(label, dst_label_file_path)

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        modalities=("cine-MRI",),
        labels={
            0: "background",
            1: "Right Ventricle Cavity",
            2: "Myocardium",
            3: "Left Ventricle Cavity",
        },
        dataset_name=task_name,
        license="",
        dataset_description="ACDC - Automated Cardiac Diagnosis Challenge",
        dataset_reference="https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html",
    )


if __name__ == "__main__":
    convert()
