import nibabel as nib
import nibabel.processing as nibpro
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subdirs
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data
from sklearn.model_selection import train_test_split


def convert(path: str, subdir: str = "ISLES-2022"):
    # INPUT DATA
    path = f"{path}/{subdir}"
    file_suffix = ".nii.gz"

    # Train/Test Splits
    images_dir = join(path, "images")
    labels_dir = join(path, "labels_derivatives")

    training_samples, test_samples = train_test_split(subdirs(labels_dir, join=False), random_state=859032)

    ###OUTPUT DATA
    # Target names
    task_name = "Task005_ISLES22_FULL"
    prefix = "ISLES22_FULL"

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
    # This is likely also the place to apply any re-orientation, resampling and/or label correction.
    for sTr in tqdm(training_samples, desc="Train"):
        label = nib.load(join(labels_dir, sTr, "ses-0001", sTr + "_ses-0001_msk" + file_suffix))

        dwi_file = nib.load(join(images_dir, sTr, "ses-0001", "dwi", sTr + "_ses-0001_dwi" + file_suffix))
        dwi_file = nibpro.resample_from_to(dwi_file, label, order=3)

        flair_file = nib.load(join(images_dir, sTr, "ses-0001", "anat", sTr + "_ses-0001_FLAIR" + file_suffix))
        flair_file = nibpro.resample_from_to(flair_file, label, order=3)

        adc_file = nib.load(join(images_dir, sTr, "ses-0001", "dwi", sTr + "_ses-0001_adc" + file_suffix))
        adc_file = nibpro.resample_from_to(adc_file, label, order=3)

        nib.save(flair_file, f"{target_imagesTr}/{prefix}_{sTr}_000.nii.gz")
        nib.save(dwi_file, f"{target_imagesTr}/{prefix}_{sTr}_001.nii.gz")
        nib.save(adc_file, f"{target_imagesTr}/{prefix}_{sTr}_002.nii.gz")
        nib.save(label, f"{target_labelsTr}/{prefix}_{sTr}.nii.gz")

    for sTs in tqdm(test_samples, desc="test"):
        label = nib.load(join(labels_dir, sTs, "ses-0001", sTs + "_ses-0001_msk" + file_suffix))

        dwi_file = nib.load(join(images_dir, sTs, "ses-0001", "dwi", sTs + "_ses-0001_dwi" + file_suffix))
        dwi_file = nibpro.resample_from_to(dwi_file, label, order=3)

        flair_file = nib.load(join(images_dir, sTs, "ses-0001", "anat", sTs + "_ses-0001_FLAIR" + file_suffix))
        flair_file = nibpro.resample_from_to(flair_file, label, order=3)

        adc_file = nib.load(join(images_dir, sTs, "ses-0001", "dwi", sTs + "_ses-0001_adc" + file_suffix))
        adc_file = nibpro.resample_from_to(adc_file, label, order=3)

        nib.save(flair_file, f"{target_imagesTs}/{prefix}_{sTs}_000.nii.gz")
        nib.save(dwi_file, f"{target_imagesTs}/{prefix}_{sTs}_001.nii.gz")
        nib.save(adc_file, f"{target_imagesTs}/{prefix}_{sTs}_002.nii.gz")
        nib.save(label, f"{target_labelsTs}/{prefix}_{sTs}.nii.gz")

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        ("Flair", "DWI", "ADC"),
        labels={0: "background", 1: "Infarct (ischemic stroke lesion)"},
        dataset_name=task_name,
        license="hands off!",
        dataset_description="ISLES-2022",
        dataset_reference="https://arxiv.org/abs/2206.06694",
    )
