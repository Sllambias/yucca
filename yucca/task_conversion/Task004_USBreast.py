"""
For this task there's 1 thing to note: images are RGB (labels are NOT).

This means each image will be shape (x, y, 3).
We cannot leave the images in this shape as the pipeline and the network will perceive this
as 3D data, which it is not.

Therefore, if we wish to preserve all the color channels, we need to save each of them as individual
files, each representing unique modalities (in this case color channels).

This would entail something like the following:
    from yucca.utils.files_and_folders import np_to_nifti_with_empty_header
   
    image = np.array(PIL.Image.open('path/to/RGB_image.png'))
    image_RED = image[:,:,0]
    image_GREEN = image[:,:,1]
    image_BLUE = image[:,:,2]

    image_RED = np_to_nifti_with_empty_header(image_RED)
    image_GREEN = np_to_nifti_with_empty_header(image_GREEN)
    image_BLUE = np_to_nifti_with_empty_header(image_BLUE)

    nib.save(image_RED, filename=f'{target_imagesTr}/{prefix}_{serial_number}_000.nii.gz')
    nib.save(image_RED, filename=f'{target_imagesTr}/{prefix}_{serial_number}_001.nii.gz')
    nib.save(image_RED, filename=f'{target_imagesTr}/{prefix}_{serial_number}_002.nii.gz')

Alternatively, as is done in this case, we convert the images to grayscale. In this dataset the
images are already grayscale and all three color channels are identical, so no information is lost.

This is achieved using:
    image = png_to_nifti('path/to/RGB_image.png', maybe_to_grayscale=True)
"""

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles,\
    subdirs
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data
from sklearn.model_selection import train_test_split
import nibabel as nib
from yucca.utils.files_and_folders import png_to_nifti
import os


folder_with_dirs = "/home/zcr545/datasets/Dataset_BUSI_with_GT"
file_suffix = '.png'
label_suffix = '_mask.png'

folders_with_images = subdirs(folder_with_dirs)

""" OUTPUT DATA
First define the task name """
task_name = 'Task004_USBreast'
prefix = 'USBreast'

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

for folder in folders_with_images:
    samples = subfiles(folder, suffix=label_suffix)
    train_samples, test_samples = train_test_split(samples, random_state=2134)

    if os.path.split(folder)[-1] == 'benign':
        l = 1
    if os.path.split(folder)[-1] == 'malignant':
        l = 2
    if os.path.split(folder)[-1] == 'normal':
        l = 0

    print(f"Using label: {l} for folder: {os.path.split(folder)[-1]}")
    for sTr in train_samples:
        sample = os.path.split(sTr[:-len(label_suffix)])[-1]
        serial_number = sample.replace('(', '').replace(')', '').split(' ')
        serial_number = serial_number[0][0] + serial_number[1]

        image_file = png_to_nifti(join(folder, sample + file_suffix), maybe_to_grayscale=True)
        label = png_to_nifti(join(folder, sample + label_suffix), maybe_to_grayscale=True,
                             is_seg=True, to_label=l)

        nib.save(image_file, filename=f'{target_imagesTr}/{prefix}_{serial_number}_000.nii.gz')
        nib.save(label, filename=f'{target_labelsTr}/{prefix}_{serial_number}.nii.gz')

    for sTs in test_samples:
        sample = os.path.split(sTs[:-len(label_suffix)])[-1]
        serial_number = sample.replace('(', '').replace(')', '').split(' ')
        serial_number = serial_number[0][0] + serial_number[1]

        image_file = png_to_nifti(join(folder, sample + file_suffix), maybe_to_grayscale=True)
        label = png_to_nifti(join(folder, sample + label_suffix), maybe_to_grayscale=True,
                             is_seg=True, to_label=l)

        nib.save(image_file, filename=f'{target_imagesTs}/{prefix}_{serial_number}_000.nii.gz')
        nib.save(label, filename=f'{target_labelsTs}/{prefix}_{serial_number}.nii.gz')

    del l

generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs,
                      modalities=('US', ),
                      labels={0: 'background', 1: 'benign', 2: 'malignant'},
                      dataset_name=task_name, license='Unknown',
                      dataset_description="Breast Ultrasound images with Segmentation masks",
                      dataset_reference="https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset?resource=download")