#%%
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles, subdirs
from yucca.task_conversion.utils import generate_dataset_json
import shutil
import gzip
from yucca.paths import yucca_raw_data
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
import os 

# INPUT DATA
# Input path and names
base_in = "/home/zcr545/projectdir/datasets/MSLesion2008"
file_suffix = '.nhdr'

# Train/Test Splits
# We only use the train folder, as the test folder does not contain segmentations 
# to obtain those, submission to the challenge is required (it's from 2008, forget it)
images_dir = labels_dir = join(base_in, 'train')

training_samples, test_samples = train_test_split(subdirs(images_dir, join=False), random_state=125896) 

###OUTPUT DATA
#Target names
task_name = 'Task041_MSLesion08'
prefix = 'MSLesion'

#Target paths
target_base = join(yucca_raw_data, task_name)

target_imagesTr = join(target_base, "imagesTr")
target_labelsTr = join(target_base, "labelsTr")

target_imagesTs = join(target_base, "imagesTs")
target_labelsTs = join(target_base, "labelsTs")

maybe_mkdir_p(target_imagesTr)
maybe_mkdir_p(target_labelsTs)
maybe_mkdir_p(target_imagesTs)
maybe_mkdir_p(target_labelsTr)

###Populate Target Directory###
for sTr in training_samples:
    image_file = sitk.ReadImage(join(images_dir, sTr, sTr+'_FLAIR'+file_suffix))
    label = sitk.ReadImage(join(labels_dir, sTr, sTr+'_lesion'+file_suffix))
    sitk.WriteImage(image_file, f'{target_imagesTr}/{prefix}_{sTr}_000.nii.gz')
    sitk.WriteImage(label, f'{target_labelsTr}/{prefix}_{sTr}.nii.gz')

for sTs in test_samples:
    image_file = sitk.ReadImage(join(images_dir, sTs, sTs+'_FLAIR'+file_suffix))
    label = sitk.ReadImage(join(labels_dir, sTs, sTs+'_lesion'+file_suffix))
    sitk.WriteImage(image_file, f'{target_imagesTs}/{prefix}_{sTs}_000.nii.gz')
    sitk.WriteImage(label, f'{target_labelsTs}/{prefix}_{sTs}.nii.gz')
                    
generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('Flair', ),
                          labels={0: 'background', 1: 'Multiple Sclerosis Lesion'},
                          dataset_name=task_name, license='hands off!',
                          dataset_description="2008 MICCAI MS Lesion Segmentation Challenge",
                          dataset_reference="https://www.nitrc.org/projects/msseg")
