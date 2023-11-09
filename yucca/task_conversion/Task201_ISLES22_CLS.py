# %%
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles, subdirs
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data
from yucca.utils.nib_utils import get_nib_orientation, reorient_nib_image
from sklearn.model_selection import train_test_split
import nibabel as nib
import nibabel.processing as nibpro
import numpy as np
import pandas as pd

# INPUT DATA
# Input path and names
base_in = "/maps/projects/image/people/zcr545/datasets/ISLES-2022"
file_suffix = ".nii.gz"

# Train/Test Splits
images_dir = join(base_in, "images")
labels_dir = join(base_in)

label_tsv = pd.read_csv(join(base_in, "participants.tsv"), sep="\t")
label_tsv = label_tsv.iloc[label_tsv["sex"].dropna().index]
label_tsv["sex"] = np.where(label_tsv["sex"] == "F", 0, 1)


training_samples, test_samples = train_test_split(label_tsv["participant_id"], random_state=859032)
tr33 = int(len(training_samples) * 0.33)
tr66 = int(len(training_samples) * 0.66)
ts33 = int(len(test_samples) * 0.33)
ts66 = int(len(test_samples) * 0.66)

###OUTPUT DATA
# Target names
task_name = "Task201_ISLES22_CLS"
prefix = "ISLES22_DWI"
tasks = ["Classification", "Reconstruction", "Segmentation"]
# Target paths
target_base = join(yucca_raw_data, task_name)

target_imagesTr = join(target_base, "imagesTr")
target_labelsTr = join(target_base, "labelsTr")

target_imagesTs = join(target_base, "imagesTs")
target_labelsTs = join(target_base, "labelsTs")

for task in tasks:
    maybe_mkdir_p(join(target_imagesTr, task))
    maybe_mkdir_p(join(target_labelsTs, task))
    maybe_mkdir_p(join(target_imagesTs, task))
    maybe_mkdir_p(join(target_labelsTr, task))

###Populate Target Directory###
# This is likely also the place to apply any re-orientation, resampling and/or label correction.
task = tasks[0]
for sTr in training_samples[:tr33]:
    image_file = nib.load(join(images_dir, sTr, "ses-0001", "dwi", sTr + "_ses-0001_dwi" + file_suffix))
    label = label_tsv[label_tsv["participant_id"] == sTr]["sex"].to_numpy()

    nib.save(image_file, f"{target_imagesTr}/{task}/{prefix}_{sTr}_000.nii.gz")
    np.save(f"{target_labelsTr}/{task}/{prefix}_{sTr}.npy", label)

for sTs in test_samples[:ts33]:
    image_file = nib.load(join(images_dir, sTs, "ses-0001", "dwi", sTs + "_ses-0001_dwi" + file_suffix))
    label = label_tsv[label_tsv["participant_id"] == sTs]["sex"].to_numpy()

    nib.save(image_file, f"{target_imagesTs}/{task}/{prefix}_{sTs}_000.nii.gz")
    np.save(f"{target_labelsTs}/{task}/{prefix}_{sTs}.npy", label)

task = tasks[1]
for sTr in training_samples[tr33:tr66]:
    image_file = nib.load(join(images_dir, sTr, "ses-0001", "dwi", sTr + "_ses-0001_dwi" + file_suffix))

    nib.save(image_file, f"{target_imagesTr}/{task}/{prefix}_{sTr}_000.nii.gz")

for sTs in test_samples[ts33:ts66]:
    image_file = nib.load(join(images_dir, sTs, "ses-0001", "dwi", sTs + "_ses-0001_dwi" + file_suffix))

    nib.save(image_file, f"{target_imagesTs}/{task}/{prefix}_{sTs}_000.nii.gz")

task = tasks[2]
labels_dir = join(base_in, "labels_derivatives")

for sTr in training_samples[tr66:]:
    image_file = nib.load(join(images_dir, sTr, "ses-0001", "dwi", sTr + "_ses-0001_dwi" + file_suffix))
    label = nib.load(join(labels_dir, sTr, "ses-0001", sTr + "_ses-0001_msk" + file_suffix))
    image_file = nibpro.resample_from_to(image_file, label, order=3)

    data = label.get_fdata()
    data[data > 0] = 1
    label = nib.Nifti1Image(data, label.affine, label.header)

    nib.save(image_file, f"{target_imagesTr}/{task}/{prefix}_{sTr}_000.nii.gz")
    nib.save(label, f"{target_labelsTr}/{task}/{prefix}_{sTr}.nii.gz")

for sTs in test_samples[ts66:]:
    image_file = nib.load(join(images_dir, sTs, "ses-0001", "dwi", sTs + "_ses-0001_dwi" + file_suffix))
    label = nib.load(join(labels_dir, sTs, "ses-0001", sTs + "_ses-0001_msk" + file_suffix))
    image_file = nibpro.resample_from_to(image_file, label, order=3)

    data = label.get_fdata()
    data[data > 0] = 1
    label = nib.Nifti1Image(data, label.affine, label.header)

    nib.save(image_file, f"{target_imagesTs}/{task}/{prefix}_{sTs}_000.nii.gz")
    nib.save(label, f"{target_labelsTs}/{task}/{prefix}_{sTs}.nii.gz")

generate_dataset_json(
    join(target_base, "dataset.json"),
    target_imagesTr,
    target_imagesTs,
    ("DWI",),
    labels={tasks[0]: {0: "Female", 1: "Male"}, tasks[1]: {}, tasks[2]: {0: "Background", 1: "Stroke"}},
    tasks=tasks,
    dataset_name=task_name,
    license="hands off!",
    dataset_description="ISLES-2022",
    dataset_reference="https://arxiv.org/abs/2206.06694",
)

# %%
