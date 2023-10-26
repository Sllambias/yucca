import nibabel as nib
import nibabel.processing as nibpro
from sklearn.model_selection import train_test_split
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data
from yucca.utils.nib_utils import get_nib_direction, reorient_nib_image

""" INPUT DATA - Define input path and suffixes """

folder_with_images = "/path/to/folder"
file_extension = '.nii'

""" OUTPUT DATA - Define the task name and prefix """
task_name = 'Task000_MyTask'
task_prefix = 'MyTask'

""" Access the input data. If images are not split into train/test, and you wish to randomly 
split the data, uncomment and adapt the following lines to fit your local path. """

#images_dir = join(folder_with_images, 'data_dir', 'images')
#labels_dir = join(folder_with_images, 'data_dir', 'labels')

#samples = subfiles(labels_dir, join=False, suffix=file_extension)
#train_samples, test_samples = train_test_split(samples, test_size = 0.2, random_state = 42154)

#images_dir_tr = images_dir_ts = images_dir
#labels_dir_tr = labels_dir_ts = labels_dir

""" If images are already split into train/test and images/labels uncomment and adapt the following 
lines to fit your local path."""

#images_dir_tr = join(folder_with_images, 'train_dir', 'images')
#labels_dir_tr = join(folder_with_images, 'train_dir', 'labels')
#train_samples = subfiles(labels_dir_tr, join=False, suffix=file_extension)

#images_dir_ts = join(folder_with_images, 'test_dir', 'images')
#labels_dir_ts = join(folder_with_images, 'test_dir', 'labels')    
#test_samples = subfiles(labels_dir_ts, join=False, suffix=file_extension)

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
    case_id = sTr[:-len(file_extension)]
    image = nib.load(join(images_dir_tr, sTr))
    label = nib.load(join(labels_dir_tr, sTr))

    # Orient to RAS and register image-label, using the image as reference.
    orig_ornt = get_nib_direction(image)
    flair_file = reorient_nib_image(image, original_orientation=orig_ornt,
                                    target_orientation='RAS')
    label = nibpro.resample_from_to(label, image)

    nib.save(image, filename=f'{target_imagesTr}/{task_prefix}_{sTr}_000.nii.gz')
    nib.save(label, filename=f'{target_labelsTr}/{task_prefix}_{sTr}.nii.gz')

for sTs in test_samples:
    case_id = sTs[:-len(file_extension)]
    image = nib.load(join(images_dir_tr, sTs))
    label = nib.load(join(labels_dir_tr, sTs))

    # Orient to RAS and register image-label, using the image as reference.
    orig_ornt = get_nib_direction(image)
    flair_file = reorient_nib_image(image, original_orientation=orig_ornt,
                                    target_orientation='RAS')
    label = nibpro.resample_from_to(label, image)

    nib.save(image, filename=f'{target_imagesTs}/{task_prefix}_{sTs}_000.nii.gz')
    nib.save(label, filename=f'{target_labelsTs}/{task_prefix}_{sTs}.nii.gz')

generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs,
                      modalities=('T1', ),
                      labels={0: 'background, probably', 1: 'Fake Label', 2: 'Also Fake Label'},
                      dataset_name=task_name, license='Template',
                      dataset_description="Template Dataset",
                      dataset_reference="Link to source or similar")