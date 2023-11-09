import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data

""" INPUT DATA - Define input path and suffixes """

folder_with_images = "/home/zcr545/datasets/BONBID2023_Train"
ext = ".mha"

adc_suffix = "-ADC_ss"
zadc_suffix = "-ADC_smooth2mm_clipped10"
zadc_prefix = "Zmap_"
label_suffix = "_lesion"


""" OUTPUT DATA - Define the task name and prefix """
task_name = "Task008_BONBID"
task_prefix = "BB"

""" Access the input data. If images are not split into train/test, and you wish to randomly 
split the data, uncomment and adapt the following lines to fit your local path. """

images_dir_0 = join(folder_with_images, "1ADC_ss")
images_dir_1 = join(folder_with_images, "2Z_ADC")

labels_dir = join(folder_with_images, "3LABEL")

samples = subfiles(labels_dir, join=False, suffix=ext)
train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=4215532)

""" Then define target paths """
target_base = join(yucca_raw_data, task_name)

target_imagesTr = join(target_base, "imagesTr")
target_labelsTr = join(target_base, "labelsTr")

target_imagesTs = join(target_base, "imagesTs")
target_labelsTs = join(target_base, "labelsTs")

maybe_mkdir_p(target_imagesTr)
maybe_mkdir_p(target_labelsTs)
maybe_mkdir_p(target_imagesTs)
maybe_mkdir_p(target_labelsTr)

"""Populate Target Directory
This is also the place to apply any re-orientation, resampling and/or label correction."""

for sTr in train_samples:
    case_id = sTr[: -len(label_suffix + ext)]
    ADC_image = sitk.ReadImage(join(images_dir_0, case_id + adc_suffix + ext))
    zADC_image = sitk.ReadImage(join(images_dir_1, zadc_prefix + case_id + zadc_suffix + ext))

    label = sitk.ReadImage(join(labels_dir, sTr))

    sitk.WriteImage(ADC_image, f"{target_imagesTr}/{task_prefix}_{case_id}_000.nii.gz")
    sitk.WriteImage(zADC_image, f"{target_imagesTr}/{task_prefix}_{case_id}_001.nii.gz")
    sitk.WriteImage(label, f"{target_labelsTr}/{task_prefix}_{case_id}.nii.gz")

for sTs in test_samples:
    case_id = sTs[: -len(label_suffix + ext)]
    ADC_image = sitk.ReadImage(join(images_dir_0, case_id + adc_suffix + ext))
    zADC_image = sitk.ReadImage(join(images_dir_1, zadc_prefix + case_id + zadc_suffix + ext))

    label = sitk.ReadImage(join(labels_dir, sTs))

    sitk.WriteImage(ADC_image, f"{target_imagesTs}/{task_prefix}_{case_id}_000.nii.gz")
    sitk.WriteImage(zADC_image, f"{target_imagesTs}/{task_prefix}_{case_id}_001.nii.gz")
    sitk.WriteImage(label, f"{target_labelsTs}/{task_prefix}_{case_id}.nii.gz")

generate_dataset_json(
    join(target_base, "dataset.json"),
    target_imagesTr,
    target_imagesTs,
    modalities=(
        "ADC",
        "zADC",
    ),
    labels={0: "background", 1: "Hypoxic Ischemic Encephalopathy (HIE)"},
    dataset_name=task_name,
    license="",
    dataset_description="BONBIE-HIE 2023",
    dataset_reference="https://bonbid-hie2023.grand-challenge.org/bonbid-hie2023/",
)
