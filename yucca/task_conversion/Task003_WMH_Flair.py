#%%
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles, subfolders
from yucca.task_conversion.utils import generate_dataset_json
from yucca.utils.nib_utils import get_nib_orientation, reorient_nib_image
import shutil
import gzip
from yucca.paths import yucca_raw_data
import nibabel as nib
import nibabel.processing as nibpro

###INPUT DATA###
#Input path and names
base_in = "/home/zcr545/datasets/WMH"
file_suffix = '.nii.gz'

datasets = ["Amsterdam", "Singapore", "Utrecht"]
site = ""

###OUTPUT DATA
#Target names
task_name = 'Task003_WMH_Flair'
prefix = 'WMH_F'

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
for dataset in datasets:
    dataset_path = join(base_in, dataset)
    if dataset == 'Amsterdam':
        tr_folder = 'Train_GE3T'
    else:
        tr_folder = 'Train'

    train_folder = join(dataset_path, tr_folder)
    test_folder = join(dataset_path, 'Test')

    training_samples = subfolders(train_folder, join=False)
    test_samples = subfolders(test_folder, join=False)

    # First we sort the training data
    for sTr in training_samples:
        # Loading relevant modalities and the ground truth
        flair_file = nib.load(join(train_folder, sTr, 'pre', 'FLAIR.nii.gz'))

        mask = nib.load(join(train_folder, sTr, 'pre', 'wmh.nii.gz'))

        # Aligning modalities and masks
        orig_ornt = get_nib_orientation(flair_file)
        flair_file = reorient_nib_image(flair_file, original_orientation=orig_ornt,
                                        target_orientation='RAS')
        mask = nibpro.resample_from_to(mask, flair_file, order=0)
        data = mask.get_fdata()
        data[data != 1] = 0
        mask = nib.Nifti1Image(data, mask.affine, mask.header)

        nib.save(flair_file, filename=f'{target_imagesTr}/{prefix}_{sTr}_000.nii.gz')
        nib.save(mask, filename=f'{target_labelsTr}/{prefix}_{sTr}.nii.gz')

    # Now we sort the test data
    if dataset == 'Amsterdam':
        for site in test_samples:
            samples = subfolders(join(test_folder, site), join=False)
            for sTs in samples:
                flair_file = nib.load(join(test_folder, site, sTs, 'pre', 'FLAIR.nii.gz'))

                mask = nib.load(join(test_folder, site, sTs, 'pre', 'wmh.nii.gz'))

                # Aligning modalities and masks
                orig_ornt = get_nib_orientation(flair_file)
                flair_file = reorient_nib_image(flair_file, original_orientation=orig_ornt,
                                                target_orientation='RAS')
                
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
            orig_ornt = get_nib_orientation(flair_file)
            flair_file = reorient_nib_image(flair_file, original_orientation=orig_ornt,
                                            target_orientation='RAS')
            
            mask = nibpro.resample_from_to(mask, flair_file, order=0)
            data = mask.get_fdata()
            data[data != 1] = 0
            mask = nib.Nifti1Image(data, mask.affine, mask.header)

            nib.save(flair_file, filename=f'{target_imagesTs}/{prefix}_{sTs}_000.nii.gz')
            nib.save(mask, filename=f'{target_labelsTs}/{prefix}_{sTs}.nii.gz')


generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('Flair', ),
                        labels={0: 'background', 1: 'WMH',}, 
                        dataset_name=task_name, license='hands off!',
                        dataset_description="OASIS Dataset",
                        dataset_reference="")