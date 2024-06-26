import shutil
from sklearn.model_selection import train_test_split
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subdirs
from yucca.pipeline.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data, yucca_source


def convert(path: str = yucca_source, subdir: str = "MBAS_Dataset"):
    # INPUT DATA
    path = f"{path}/{subdir}"

    # Train/Test Splits
    images_dir = join(path, "Training")
    training_samples, test_samples = train_test_split(subdirs(images_dir, join=False), random_state=35219)

    ###OUTPUT DATA
    # Target names
    task_name = "Task014_MBAS24"
    task_prefix = "MBAS24"

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

    for sTr in training_samples:
        src_image_file_path = join(images_dir, sTr, sTr + "_gt.nii.gz")
        src_label_file_path = join(images_dir, sTr, sTr + "_label.nii.gz")
        dst_image_file_path = f"{target_imagesTr}/{task_prefix}_{sTr}_000.nii.gz"

        dst_label_path = f"{target_labelsTr}/{task_prefix}_{sTr}.nii.gz"
        shutil.copy2(src_image_file_path, dst_image_file_path)
        shutil.copy2(src_label_file_path, dst_label_path)

    del sTr
    for sTs in test_samples:
        src_image_file_path = join(images_dir, sTs, sTs + "_gt.nii.gz")
        src_label_file_path = join(images_dir, sTs, sTs + "_label.nii.gz")
        dst_image_file_path = f"{target_imagesTs}/{task_prefix}_{sTs}_000.nii.gz"

        dst_label_path = f"{target_labelsTs}/{task_prefix}_{sTs}.nii.gz"
        shutil.copy2(src_image_file_path, dst_image_file_path)
        shutil.copy2(src_label_file_path, dst_label_path)

    del sTs
    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        modalities=("MRI (LGE)",),
        labels={0: "background", 1: "Right Atrium Cavity", 2: "Left Atrium Cavity", 3: "Left & Right Atrium Wall"},
        dataset_name=task_name,
        license="CC-BY-SA 4.0",
        dataset_description="MICCAI MBAS 2024",
        dataset_reference="https://codalab.lisn.upsaclay.fr/competitions/18516#learn_the_details",
    )


if __name__ == "__main__":
    convert()
