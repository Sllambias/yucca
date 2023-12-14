import nibabel as nib
import nibabel.processing as nibpro
from sklearn.model_selection import train_test_split
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data
from yucca.utils.nib_utils import get_nib_direction, reorient_nib_image

# INPUT DATA
# Define input path and extension

folder_with_images = "/home/zcr545/datasets/decathlon/Task09_Spleen"

file_extension = ".nii.gz"

# OUTPUT DATA
# Define the task name and prefix
task_name = "Task029_Spleen"
task_prefix = "Spleen"

# Set target paths
target_base = join(yucca_raw_data, task_name)
target_imagesTr = join(target_base, "imagesTr")
target_labelsTr = join(target_base, "labelsTr")
target_imagesTs = join(target_base, "imagesTs")
target_labelsTs = join(target_base, "labelsTs")

maybe_mkdir_p(target_imagesTr)
maybe_mkdir_p(target_labelsTs)
maybe_mkdir_p(target_imagesTs)
maybe_mkdir_p(target_labelsTr)

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
    image = nib.load(join(images_dir_tr, sTr))
    label = nib.load(join(labels_dir_tr, sTr))
    sTr = sTr[: -len(file_extension)]

    # Orient to RAS and register image-label, using the image as reference.
    orig_ornt = get_nib_direction(image)
    image = reorient_nib_image(image, original_orientation=orig_ornt, target_orientation="RAS")

    label = nibpro.resample_from_to(label, image, order=0)

    nib.save(image, filename=f"{target_imagesTr}/{task_prefix}_{sTr}_000.nii.gz")
    nib.save(label, filename=f"{target_labelsTr}/{task_prefix}_{sTr}.nii.gz")

for sTs in test_samples:
    image = nib.load(join(images_dir_ts, sTs))
    label = nib.load(join(labels_dir_ts, sTs))
    sTs = sTs[: -len(file_extension)]

    # Orient to RAS and register image-label, using the image as reference.
    orig_ornt = get_nib_direction(image)
    image = reorient_nib_image(image, original_orientation=orig_ornt, target_orientation="RAS")

    label = nibpro.resample_from_to(label, image, order=0)

    nib.save(image, filename=f"{target_imagesTs}/{task_prefix}_{sTs}_000.nii.gz")
    nib.save(label, filename=f"{target_labelsTs}/{task_prefix}_{sTs}.nii.gz")

generate_dataset_json(
    join(target_base, "dataset.json"),
    target_imagesTr,
    target_imagesTs,
    modalities=("CT",),
    labels={0: "Background", 1: "Spleen"},
    dataset_name=task_name,
    license="CC-BY-SA 4.0",
    dataset_description="Decathlon: Spleen Segmentation",
    dataset_reference="Memorial Sloan Kettering Cancer Center",
)
