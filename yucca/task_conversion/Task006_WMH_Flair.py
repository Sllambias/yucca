import shutil
import gzip
from sklearn.model_selection import train_test_split
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfolders
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data


def convert(path: str, subdir: str = "WMH"):
    """INPUT DATA - Define input path and suffixes"""
    path = join(path, subdir)
    file_suffix = ".nii.gz"

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task006_WMH_Flair"
    task_prefix = "WMH_F"

    datasets = ["Amsterdam", "Singapore", "Utrecht"]
    site = ""


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

    ###Populate Target Directory###
    for dataset in datasets:
        dataset_path = join(path, dataset)
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
            image_file = open(join(train_folder, sTr, 'pre', 'FLAIR.nii.gz'), "rb")
            label = open(join(train_folder, sTr, 'pre', 'wmh.nii.gz'), "rb")
            shutil.copyfileobj(image_file, gzip.open(f"{target_imagesTr}/{task_prefix}_{sTr}_000.nii.gz", "wb"))
            shutil.copyfileobj(label, gzip.open(f"{target_labelsTr}/{task_prefix}_{sTr}.nii.gz", "wb"))

        # Now we sort the test data
        if dataset == 'Amsterdam':
            for site in test_samples:
                samples = subfolders(join(test_folder, site), join=False)
                for sTs in samples:
                    image_file = open(join(test_folder, site, sTs, 'pre', 'FLAIR.nii.gz'), "rb")
                    label = open(join(test_folder, site, sTs, 'pre', 'wmh.nii.gz'), "rb")
                    shutil.copyfileobj(image_file, gzip.open(f"{target_imagesTs}/{task_prefix}_{sTs}_000.nii.gz", "wb"))
                    shutil.copyfileobj(label, gzip.open(f"{target_labelsTs}/{task_prefix}_{sTs}.nii.gz", "wb"))
        else:
            for sTs in test_samples:
                image_file = open(join(test_folder, sTs, 'pre', 'FLAIR.nii.gz'), "rb")
                label = open(join(test_folder, sTs, 'pre', 'wmh.nii.gz'), "rb")
                shutil.copyfileobj(image_file, gzip.open(f"{target_imagesTs}/{task_prefix}_{sTs}_000.nii.gz", "wb"))
                shutil.copyfileobj(label, gzip.open(f"{target_labelsTs}/{task_prefix}_{sTs}.nii.gz", "wb"))

    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('Flair'),
                        labels={0: 'background', 1: 'WMH', 2: 'Other Pathology'},
                        dataset_name=task_name, license='CC BY-NC 4.0 DEED',
                        dataset_description="White Matter Hyperintensity Segmentation Challenge. Flair images only!",
                        dataset_reference="https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/AECRSD")
