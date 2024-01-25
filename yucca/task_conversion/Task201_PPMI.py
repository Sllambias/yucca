from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.task_conversion.utils import generate_dataset_json, dirs_in_dir, should_use_volume
from yucca.paths import yucca_raw_data
from datetime import datetime
from tqdm import tqdm
import nibabel as nib


accept_modalities = ["t1", "t2", "mprage", "flair", "gre", "dwi", "swi", "grappa"]
rejec_modalities = [
    "resting_state",
    "fmri",
    "corrected",
    "mt",
    "repeat",
    "space",
]


def should_skip(modality: str):
    modality = modality.lower()
    if "2d" in modality:
        return True
    for reject in rejec_modalities:
        if reject in modality:
            return True
    for accept in accept_modalities:
        if accept in modality:
            return False
    return True


def convert(path: str, subdir: str = "PPMI"):
    """INPUT DATA - Define input path and suffixes"""
    path = join(path, subdir)

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task201_PPMI"
    task_prefix = "PPMI"

    subjects_dir = join(path, "DATA")
    target_base = join(yucca_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")

    maybe_mkdir_p(target_imagesTr)

    for subject in tqdm(dirs_in_dir(subjects_dir), desc="Subject"):
        subject_dir = join(subjects_dir, subject)
        for modality in dirs_in_dir(subject_dir):
            modality_dir = join(subject_dir, modality)
            if should_skip(modality):
                continue  # skip this modality
            for date in dirs_in_dir(modality_dir):
                date_dir = join(modality_dir, date)
                parsed_date = datetime.strptime(date, "%Y-%m-%d_%H_%M_%S.%f")
                date_simple = parsed_date.strftime("%Y%m%d")
                for image in dirs_in_dir(date_dir):
                    image_dir = join(date_dir, image)
                    if "nifti" and "dicom" in dirs_in_dir(image_dir):
                        image_dir = join(image_dir, "nifti")
                    for file in subfiles(image_dir, join=False, suffix=".nii"):
                        image_path = join(image_dir, file)
                        vol = nib.load(image_path)
                        if should_use_volume(vol):
                            output_name = f"{task_prefix}_{subject}_{modality}_{date_simple}_{image}_000.nii.gz"
                            output_path = join(target_imagesTr, output_name)
                            nib.save(vol, filename=output_path)

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        imagesTs_dir=None,
        modalities=["MRI"],
        labels=None,
        dataset_name=task_name,
        license="https://www.ppmi-info.org/sites/default/files/docs/ppmi-data-use-agreement.pdf",
        dataset_description="Parkinson Progression Markers Initiative",
        dataset_reference="https://www.ppmi-info.org/",
    )
