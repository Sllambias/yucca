#%%
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subdirs, subfiles
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data
from sklearn.model_selection import train_test_split
import nibabel as nib
import SimpleITK as sitk
import nibabel.processing as nibpro
import numpy as np
np.random.seed(512514)

# Target names
task_name = 'Task059_3BL3LV2'
prefix = '3BL3LV2'

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

###Populate Target Directory###
#This is likely also the place to apply any re-orientation, resampling and/or label correction.
for sTr in training_samples:
    image_file = nib.load(join(images_dir, sTr, 'ses-0001', 'anat', sTr+'_ses-0001_FLAIR'+file_suffix))
    label = nib.load(join(labels_dir, sTr, 'ses-0001', sTr+'_ses-0001_msk'+file_suffix))
    image_file = nibpro.resample_from_to(image_file, label, order=3)

    data = label.get_fdata()
    data[data > 0] = 2
    label = nib.Nifti1Image(data, label.affine, label.header)

    nib.save(image_file, f'{target_imagesTr}/{prefix}_{sTr}_000.nii.gz')
    nib.save(label, f'{target_labelsTr}/{prefix}_{sTr}.nii.gz')

for sTs in test_samples:
    image_file = nib.load(join(images_dir, sTs, 'ses-0001', 'anat', sTs+'_ses-0001_FLAIR'+file_suffix))
    label = nib.load(join(labels_dir, sTs, 'ses-0001', sTs+'_ses-0001_msk'+file_suffix))
    image_file = nibpro.resample_from_to(image_file, label, order=3)

    data = label.get_fdata()
    data[data > 0] = 2
    label = nib.Nifti1Image(data, label.affine, label.header)

    nib.save(image_file, f'{target_imagesTs}/{prefix}_{sTs}_000.nii.gz')
    nib.save(label, f'{target_labelsTs}/{prefix}_{sTs}.nii.gz')


# INPUT DATA
# Input path and names
base_in = "/maps/projects/image/people/zcr545/datasets/open_ms_data/cross_sectional/coregistered"
file_suffix = '.nii.gz'

# Train/Test Splits
# We only use the train folder, as the test folder does not contain segmentations 
# to obtain those, submission to the challenge is required (it's from 2008, forget it)
images_dir = labels_dir = base_in

training_samples, test_samples = train_test_split(subdirs(images_dir, join=False), random_state=125896) 

###Populate Target Directory###
for sTr in training_samples:
    image_file = nib.load(join(images_dir, sTr, 'FLAIR'+file_suffix))
    label = nib.load(join(labels_dir, sTr, 'consensus_gt'+file_suffix))
    image_file = nibpro.resample_from_to(image_file, label, order=3)
    nib.save(image_file, f'{target_imagesTr}/{prefix}_{sTr}_000.nii.gz')
    nib.save(label, f'{target_labelsTr}/{prefix}_{sTr}.nii.gz')

for sTs in test_samples:
    image_file = nib.load(join(images_dir, sTs, 'FLAIR'+file_suffix))
    label = nib.load(join(labels_dir, sTs, 'consensus_gt'+file_suffix))
    image_file = nibpro.resample_from_to(image_file, label, order=3)
    nib.save(image_file, f'{target_imagesTs}/{prefix}_{sTs}_000.nii.gz')
    nib.save(label, f'{target_labelsTs}/{prefix}_{sTs}.nii.gz')


# MSLESION_Longitudinal2015
# INPUT DATA
# Input path and names
base_in = '/maps/projects/image/people/zcr545/datasets/MSLesion_Longitudinal2015/training'
file_suffix = '.nii'

# Train/Test Splits
images_dir = labels_dir = base_in

training_samples, test_samples = train_test_split(subdirs(images_dir, join=False), random_state=4213) 

###Populate Target Directory###
for sTr in training_samples:
    for time_point in subfiles(join(images_dir, sTr, 'masks'), suffix="_mask1.nii", join=False):
        time_point = time_point[:-len("_mask1.nii")]
        image_file = nib.load(join(images_dir, sTr, 'preprocessed', time_point+'_flair_pp'+file_suffix))

        label1 = nib.load(join(labels_dir, sTr, 'masks', time_point + '_mask1' + file_suffix))
        label2 = nib.load(join(labels_dir, sTr, 'masks', time_point + '_mask2' + file_suffix))

        image_file = nibpro.resample_from_to(image_file, label1, order=3)

        data = label1.get_fdata()
        data[data > 0] = 1
        label1 = nib.Nifti1Image(data, label1.affine, label1.header)

        data = label2.get_fdata()
        data[data > 0] = 1
        label2 = nib.Nifti1Image(data, label2.affine, label2.header)

        nib.save(image_file, f'{target_imagesTr}/{prefix}_{time_point}_1_000.nii.gz')
        nib.save(image_file, f'{target_imagesTr}/{prefix}_{time_point}_2_000.nii.gz')

        nib.save(label1, f'{target_labelsTr}/{prefix}_{time_point}_1.nii.gz')
        nib.save(label2, f'{target_labelsTr}/{prefix}_{time_point}_2.nii.gz')

for sTs in test_samples:
    for time_point in subfiles(join(images_dir, sTs, 'masks'), suffix="_mask1.nii", join=False):
        time_point = time_point[:-len("_mask1.nii")]
        image_file = nib.load(join(images_dir, sTs, 'preprocessed', time_point+'_flair_pp'+file_suffix))

        label1 = nib.load(join(labels_dir, sTs, 'masks', time_point + '_mask1' + file_suffix))
        label2 = nib.load(join(labels_dir, sTs, 'masks', time_point + '_mask2' + file_suffix))

        image_file = nibpro.resample_from_to(image_file, label1, order=3)

        data = label1.get_fdata()
        data[data > 0] = 1
        label1 = nib.Nifti1Image(data, label1.affine, label1.header)

        data = label2.get_fdata()
        data[data > 0] = 1
        label2 = nib.Nifti1Image(data, label2.affine, label2.header)

        nib.save(image_file, f'{target_imagesTs}/{prefix}_{time_point}_1_000.nii.gz')
        nib.save(image_file, f'{target_imagesTs}/{prefix}_{time_point}_2_000.nii.gz')

        nib.save(label1, f'{target_labelsTs}/{prefix}_{time_point}_1.nii.gz')
        nib.save(label2, f'{target_labelsTs}/{prefix}_{time_point}_2.nii.gz')




# MSSEG 2016
# INPUT DATA
# Input path and names
base_in = "/maps/projects/image/people/zcr545/datasets/MSSEG1_2016"
file_suffix = '.nii.gz'

# Train/Test Splits
train_folder = join(base_in, "training")
test_folder = join(base_in, "Testing")


###Populate Target Directory###
for center in subdirs(train_folder, join=False):
    training_samples = subdirs(join(train_folder, center), join=False)
    for sTr in training_samples:
        image_file = nib.load(join(train_folder, center, sTr, "Preprocessed_Data", 'FLAIR_preprocessed'+file_suffix))
        label = nib.load(join(train_folder, center, sTr, "Masks", 'Consensus'+file_suffix))

        data = label.get_fdata()
        data[data > 0] = 1
        label = nib.Nifti1Image(data, label.affine, label.header)

        nib.save(image_file, f'{target_imagesTr}/{prefix}_{center+sTr}_000.nii.gz')
        nib.save(label, f'{target_labelsTr}/{prefix}_{center+sTr}.nii.gz')

for center in subdirs(test_folder, join=False):
    test_samples = subdirs(join(test_folder, center), join=False)
    for sTs in test_samples:
        image_file = nib.load(join(test_folder, center,  sTs, "Preprocessed_Data", 'FLAIR_preprocessed'+file_suffix))
        label = nib.load(join(test_folder, center, sTs, "Masks", 'Consensus'+file_suffix))

        data = label.get_fdata()
        data[data > 0] = 1
        label = nib.Nifti1Image(data, label.affine, label.header)

        nib.save(image_file, f'{target_imagesTs}/{center+prefix}_{sTs}_000.nii.gz')
        nib.save(label, f'{target_labelsTs}/{center+prefix}_{sTs}.nii.gz')


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
        tr_folder = 'Train_GE3T'
    else:
        tr_folder = 'Train'

    train_folder = join(dataset_path, tr_folder)
    test_folder = join(dataset_path, 'Test')

    training_samples = subdirs(train_folder, join=False)
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
        data[data == 1] = 3
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
                data[data == 1] = 3
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
            data[data == 1] = 3
            mask = nib.Nifti1Image(data, mask.affine, mask.header)

            nib.save(flair_file, filename=f'{target_imagesTs}/{prefix}_{sTs}_000.nii.gz')
            nib.save(mask, filename=f'{target_labelsTs}/{prefix}_{sTs}.nii.gz')

hierarchy = {
    "Background":0,
    "Lesion":{
        "WM":{
            "WMH":3,
            "MS":1},
        "Stroke":2
        }
    }

generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('Flair', ),
                      labels={0: 'background', 1: 'MS Lesion', 2: 'Stroke', 3: 'WMH'}, label_hierarchy=hierarchy,
                      dataset_name=task_name, license='hands off!',
                      dataset_description="MSLesion 2008 and ISLES-2022 and WMH",
                      dataset_reference="https://www.nitrc.org/projects/msseg, https://arxiv.org/abs/2206.06694")

# %%
