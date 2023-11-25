import shutil
import gzip
from sklearn.model_selection import train_test_split
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data


def convert(path: str, subdir: str = "MyDataset"):
    """INPUT DATA - Define input path and suffixes"""
    path = join(path, subdir)
    file_suffix = ".nii"

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task000_MyTask"
    task_prefix = "MyTask"

    """ Access the input data. If images are not split into train/test, and you wish to randomly 
    split the data, uncomment and adapt the following lines to fit your local path. """

    images_dir = join(path, "data_dir", "images")
    labels_dir = join(path, "data_dir", "labels")

    samples = subfiles(labels_dir, join=False, suffix=file_suffix)
    train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42154)

    images_dir_tr = images_dir_ts = images_dir
    labels_dir_tr = labels_dir_ts = labels_dir

    """ If images are already split into train/test and images/labels uncomment and adapt the following 
    lines to fit your local path."""

    # images_dir_tr = join(path, 'train_dir', 'images')
    # labels_dir_tr = join(path, 'train_dir', 'labels')
    # train_samples = subfiles(labels_dir_tr, join=False, suffix=file_suffix)

    # images_dir_ts = join(path, 'test_dir', 'images')
    # labels_dir_ts = join(path, 'test_dir', 'labels')
    # test_samples = subfiles(labels_dir_ts, join=False, suffix=file_suffix)

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
        case_id = sTr[: -len(file_suffix)]
        image_file = open(join(images_dir_tr, sTr), "rb")
        label = open(join(labels_dir_tr, sTr), "rb")
        shutil.copyfileobj(image_file, gzip.open(f"{target_imagesTr}/{task_prefix}_{case_id}_000.nii.gz", "wb"))
        shutil.copyfileobj(label, gzip.open(f"{target_labelsTr}/{task_prefix}_{case_id}.nii.gz", "wb"))

    for sTs in test_samples:
        case_id = sTs[: -len(file_suffix)]
        image_file = open(join(images_dir_ts, sTs), "rb")
        label = open(join(labels_dir_ts, sTs), "rb")
        shutil.copyfileobj(image_file, gzip.open(f"{target_imagesTs}/{task_prefix}_{case_id}_000.nii.gz", "wb"))
        shutil.copyfileobj(label, gzip.open(f"{target_labelsTs}/{task_prefix}_{case_id}.nii.gz", "wb"))

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        modalities=("T1",),
        labels={0: "background, probably", 1: "Fake Label", 2: "Also Fake Label"},
        dataset_name=task_name,
        license="Template",
        dataset_description="Template Dataset",
        dataset_reference="Link to source or similar",
    )
