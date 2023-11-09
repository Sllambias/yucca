# %%
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subdirs
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data
from sklearn.model_selection import train_test_split
import nibabel as nib
import SimpleITK as sitk
import nibabel.processing as nibpro
import numpy as np

np.random.seed(512514)

# Target names
task_name = "Task054_MSSeg"
prefix = "MSSeg"

###OUTPUT DATA
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

# MSSEG 2016
# INPUT DATA
# Input path and names
base_in = "/maps/projects/image/people/zcr545/datasets/MSSEG1_2016"
file_suffix = ".nii.gz"

# Train/Test Splits
# We only use the train folder, as the test folder does not contain segmentations
# to obtain those, submission to the challenge is required (it's from 2008, forget it)
train_folder = join(base_in, "training")
test_folder = join(base_in, "Testing")


###Populate Target Directory###
for center in subdirs(train_folder, join=False):
    training_samples = subdirs(join(train_folder, center), join=False)
    for sTr in training_samples:
        image_file = sitk.ReadImage(join(train_folder, center, sTr, "Preprocessed_Data", "FLAIR_preprocessed" + file_suffix))
        label = sitk.ReadImage(join(train_folder, center, sTr, "Masks", "Consensus" + file_suffix))
        sitk.WriteImage(image_file, f"{target_imagesTr}/{prefix}_{center+sTr}_000.nii.gz")
        sitk.WriteImage(label, f"{target_labelsTr}/{prefix}_{center+sTr}.nii.gz")

for center in subdirs(test_folder, join=False):
    test_samples = subdirs(join(test_folder, center), join=False)
    for sTs in test_samples:
        image_file = sitk.ReadImage(join(test_folder, center, sTs, "Preprocessed_Data", "FLAIR_preprocessed" + file_suffix))
        label = sitk.ReadImage(join(test_folder, center, sTs, "Masks", "Consensus" + file_suffix))
        sitk.WriteImage(image_file, f"{target_imagesTs}/{center+prefix}_{sTs}_000.nii.gz")
        sitk.WriteImage(label, f"{target_labelsTs}/{center+prefix}_{sTs}.nii.gz")

generate_dataset_json(
    join(target_base, "dataset.json"),
    target_imagesTr,
    target_imagesTs,
    ("Flair",),
    labels={0: "background", 1: "MS Lesion"},
    dataset_name=task_name,
    license="hands off!",
    dataset_description="MSLesion 2008 and ISLES-2022 and WMH",
    dataset_reference="https://www.nitrc.org/projects/msseg, https://arxiv.org/abs/2206.06694",
)
