#%%
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
task_name = 'Task051_3BrainLesionFT'
prefix = '3BLFT'

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


# INPUT DATA
# Input path and names
base_in = "/maps/projects/image/people/zcr545/datasets/ISLES-2022"
file_suffix = '.nii.gz'

# Train/Test Splits
images_dir = join(base_in, 'images')
labels_dir = join(base_in, 'labels_derivatives')

training_samples, test_samples = train_test_split(subdirs(labels_dir, join=False), random_state=859032)
training_samples = training_samples[:14]

###Populate Target Directory###
#This is likely also the place to apply any re-orientation, resampling and/or label correction.
for sTr in training_samples:
    image_file = nib.load(join(images_dir, sTr, 'ses-0001', 'anat', sTr+'_ses-0001_FLAIR'+file_suffix))
    label = nib.load(join(labels_dir, sTr, 'ses-0001', sTr+'_ses-0001_msk'+file_suffix))
    label = nibpro.resample_from_to(label, image_file, order=0)

    nib.save(image_file, f'{target_imagesTr}/{prefix}_{sTr}_000.nii.gz')
    nib.save(label, f'{target_labelsTr}/{prefix}_{sTr}.nii.gz')

for sTs in test_samples:
    image_file = nib.load(join(images_dir, sTs, 'ses-0001', 'anat', sTs+'_ses-0001_FLAIR'+file_suffix))
    label = nib.load(join(labels_dir, sTs, 'ses-0001', sTs+'_ses-0001_msk'+file_suffix))
    label = nibpro.resample_from_to(label, image_file, order=0)

    nib.save(image_file, f'{target_imagesTs}/{prefix}_{sTs}_000.nii.gz')
    nib.save(label, f'{target_labelsTs}/{prefix}_{sTs}.nii.gz')


# INPUT DATA
# Input path and names
base_in = "/maps/projects/image/people/zcr545/datasets/MSLesion2008"
file_suffix = '.nhdr'

# Train/Test Splits
# We only use the train folder, as the test folder does not contain segmentations 
# to obtain those, submission to the challenge is required (it's from 2008, forget it)
images_dir = labels_dir = join(base_in, 'train')

training_samples, test_samples = train_test_split(subdirs(images_dir, join=False), random_state=125896) 

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


###INPUT DATA###
#Input path and names
base_in = "/maps/projects/image/people/zcr545/datasets/WMH"
file_suffix = '.nii.gz'

datasets = ["Amsterdam", "Singapore", "Utrecht"]
site = ""

###Populate Target Directory###
for dataset in datasets:
    dataset_path = join(base_in, dataset)
    if dataset == 'Amsterdam':
        s = 4
        tr_folder = 'Train_GE3T'
    else:
        s = 5
        tr_folder = 'Train'

    train_folder = join(dataset_path, tr_folder)
    test_folder = join(dataset_path, 'Test')

    training_samples = subdirs(train_folder, join=False)
    np.random.shuffle(training_samples)
    training_samples = training_samples[:s]

    test_samples = subdirs(test_folder, join=False)

    # First we sort the training data
    for sTr in training_samples:
        # Loading relevant modalities and the ground truth
        flair_file = nib.load(join(train_folder, sTr, 'pre', 'FLAIR.nii.gz'))

        mask = nib.load(join(train_folder, sTr, 'pre', 'wmh.nii.gz'))

        # Aligning modalities and masks
        mask = nibpro.resample_from_to(mask, flair_file, order=0)
        data = mask.get_fdata()
        data[data != 1] = 0
        mask = nib.Nifti1Image(data, mask.affine, mask.header)

        nib.save(flair_file, filename=f'{target_imagesTr}/{prefix}_{sTr}_000.nii.gz')
        nib.save(mask, filename=f'{target_labelsTr}/{prefix}_{sTr}.nii.gz')

    # Now we sort the test data
    if dataset == 'Amsterdam':
        for site in test_samples:
            samples = subdirs(join(test_folder, site), join=False)
            for sTs in samples:
                flair_file = nib.load(join(test_folder, site, sTs, 'pre', 'FLAIR.nii.gz'))

                mask = nib.load(join(test_folder, site, sTs, 'pre', 'wmh.nii.gz'))

                # Aligning modalities and masks
                mask = nibpro.resample_from_to(mask, flair_file, order=0)

                data = mask.get_fdata()
                data[data != 1] = 0
                mask = nib.Nifti1Image(data, mask.affine, mask.header)

                nib.save(flair_file, filename=f'{target_imagesTs}/{prefix}_{sTs}_000.nii.gz')
                nib.save(mask, filename=f'{target_labelsTs}/{prefix}_{sTs}.nii.gz')
    else:
        for sTs in test_samples:
            flair_file = nib.load(join(test_folder, sTs, 'pre', 'FLAIR.nii.gz'))

            mask = nib.load(join(test_folder, sTs, 'pre', 'wmh.nii.gz'))

            # Aligning modalities and masks
            mask = nibpro.resample_from_to(mask, flair_file, order=0)

            data = mask.get_fdata()
            data[data != 1] = 0
            mask = nib.Nifti1Image(data, mask.affine, mask.header)

            nib.save(flair_file, filename=f'{target_imagesTs}/{prefix}_{sTs}_000.nii.gz')
            nib.save(mask, filename=f'{target_labelsTs}/{prefix}_{sTs}.nii.gz')


generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('Flair', ),
                      labels={0: 'background', 1: 'Lesion'},
                      dataset_name=task_name, license='hands off!',
                      dataset_description="MSLesion 2008 and ISLES-2022 and WMH",
                      dataset_reference="https://www.nitrc.org/projects/msseg, https://arxiv.org/abs/2206.06694")
