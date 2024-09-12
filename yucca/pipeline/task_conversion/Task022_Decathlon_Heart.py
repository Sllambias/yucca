import nibabel as nib
import shutil
from sklearn.model_selection import train_test_split
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p as ensure_dir_exists, subfiles
from yucca.pipeline.task_conversion.utils import generate_dataset_json
from yucca.paths import get_raw_data_path
from yucca.functional.testing.data.nifti import verify_spacing_is_equal, verify_orientation_is_equal

# INPUT DATA
# Define input path and extension

folder_with_images = "/home/zcr545/datasets/decathlon/Task02_Heart"
file_extension = ".nii.gz"

# OUTPUT DATA
# Define the task name and prefix
task_name = "Task022_Heart"
task_prefix = "Heart"

# Set target paths
target_base = join(get_raw_data_path(), task_name)
target_imagesTr = join(target_base, "imagesTr")
target_labelsTr = join(target_base, "labelsTr")
target_imagesTs = join(target_base, "imagesTs")
target_labelsTs = join(target_base, "labelsTs")

ensure_dir_exists(target_imagesTr)
ensure_dir_exists(target_labelsTs)
ensure_dir_exists(target_imagesTs)
ensure_dir_exists(target_labelsTr)

# Split data
images_dir = join(folder_with_images, "imagesTr")
labels_dir = join(folder_with_images, "labelsTr")
samples = subfiles(labels_dir, join=False, suffix=file_extension)
train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=1243)
images_dir_tr = images_dir_ts = images_dir
labels_dir_tr = labels_dir_ts = labels_dir


# Populate Target Directory
# This is also the place to apply any re-orientation, resampling and/or label correction.

for sTr in train_samples:
    image_path = join(images_dir_tr, sTr)
    label_path = join(labels_dir_tr, sTr)
    sTr = sTr[: -len(file_extension)]

    image = nib.load(image_path)
    label = nib.load(label_path)
    verify_spacing_is_equal(image, label)
    verify_orientation_is_equal(image, label)

    shutil.copy2(image_path, filename=f"{target_imagesTr}/{task_prefix}_{sTr}_000.nii.gz")
    shutil.copy2(label_path, filename=f"{target_labelsTr}/{task_prefix}_{sTr}.nii.gz")

for sTs in test_samples:
    image_path = join(images_dir_ts, sTs)
    label_path = join(labels_dir_ts, sTs)
    sTs = sTs[: -len(file_extension)]

    image = nib.load(image_path)
    label = nib.load(label_path)
    verify_spacing_is_equal(image, label)
    verify_orientation_is_equal(image, label)

    shutil.copy2(image_path, filename=f"{target_imagesTs}/{task_prefix}_{sTs}_000.nii.gz")
    shutil.copy2(label_path, filename=f"{target_labelsTs}/{task_prefix}_{sTs}.nii.gz")

generate_dataset_json(
    join(target_base, "dataset.json"),
    target_imagesTr,
    target_imagesTs,
    modalities=("T1",),
    labels={0: "Background", 1: "Left Atrium"},
    dataset_name=task_name,
    license="CC-BY-SA 4.0",
    dataset_description="Decathlon: Left Atrium Segmentation",
    dataset_reference="King's College London",
)
