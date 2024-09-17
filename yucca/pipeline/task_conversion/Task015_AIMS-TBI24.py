import shutil
import nibabel as nib
from sklearn.model_selection import train_test_split
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p as ensure_dir_exists, subfiles
from yucca.pipeline.task_conversion.utils import generate_dataset_json
from yucca.paths import get_raw_data_path, get_source_path


def convert(path: str = get_source_path(), subdir: str = "AIMS-TBI24"):
    # INPUT DATA
    path = f"{path}/{subdir}"

    # Train/Test Splits
    images_dir = labels_dir = path
    samples = [i[: -len("_Lesion.nii.gz")] for i in subfiles(labels_dir, suffix="_Lesion.nii.gz", join=False)]
    training_samples, test_samples = train_test_split(samples, random_state=35219)

    ###OUTPUT DATA
    # Target names
    task_name = "Task015_AIMS-TBI24"
    task_prefix = "TBI24"

    # Target paths
    target_base = join(get_raw_data_path(), task_name)

    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")

    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")

    ensure_dir_exists(target_imagesTr)
    ensure_dir_exists(target_labelsTs)
    ensure_dir_exists(target_imagesTs)
    ensure_dir_exists(target_labelsTr)

    for sTr in training_samples:
        src_image_file_path = join(images_dir, sTr + "_T1.nii.gz")
        src_label_file_path = join(images_dir, sTr + "_Lesion.nii.gz")

        label = nib.load(src_label_file_path)
        labelarr = label.get_fdata()
        labelarr[labelarr > 0] = 1
        label = nib.Nifti1Image(labelarr, label.affine, label.header)

        dst_image_file_path = f"{target_imagesTr}/{task_prefix}_{sTr}_000.nii.gz"
        dst_label_path = f"{target_labelsTr}/{task_prefix}_{sTr}.nii.gz"

        shutil.copy2(src_image_file_path, dst_image_file_path)
        nib.save(label, dst_label_path)

    del sTr
    for sTs in test_samples:
        src_image_file_path = join(images_dir, sTs + "_T1.nii.gz")
        src_label_file_path = join(images_dir, sTs + "_Lesion.nii.gz")

        label = nib.load(src_label_file_path)
        labelarr = label.get_fdata()
        labelarr[labelarr > 0] = 1
        label = nib.Nifti1Image(labelarr, label.affine, label.header)

        dst_image_file_path = f"{target_imagesTs}/{task_prefix}_{sTs}_000.nii.gz"
        dst_label_path = f"{target_labelsTs}/{task_prefix}_{sTs}.nii.gz"

        shutil.copy2(src_image_file_path, dst_image_file_path)
        nib.save(label, dst_label_path)

    del sTs
    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        modalities=("MRI",),
        labels={
            0: "background",
            1: "Lesion",
        },
        dataset_name=task_name,
        license="CC-BY-SA 4.0",
        dataset_description="MICCAI AIMS-TBI 2024",
        dataset_reference="https://aims-tbi.grand-challenge.org/",
    )


if __name__ == "__main__":
    convert()
