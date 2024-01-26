"""
Task_conversion file for a dataset with 1 modality.

3 ways of Defining Train/Test splits:
  (1) Train-Test splits are defined by the folder structures of the data.

  (2) Splits are retrieved from a predefined .txt/.json/.pkl file:

    splits_file = 'path/to/data_splits.txt'
    with open(splits_file) as f:
      data = f.read()
    splits = json.loads(data)

  (3) Splits are defined using train_test_split from sklearn.model_selection.
  Always do this with a specified random seed.

2 ways of Defining Train/Val splits:

  (1) Random Train/Val splits. In this case nothing needs to be done, as these are defined at training time.

  (2) Predefined Train/Val according to a list of filenames.
  This list of files need to be defined and populated now.
  It can be generated using one of the resources mentioned above.

  It should be saved in the yucca_preprocessing_dir:

    from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_pickle

    maybe_mkdir_p(join(yucca_preprocessing_dir, task_name))
    save_pickle(splits, join(yucca_preprocessing_dir, task_name, 'splits.pkl'))
"""

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.task_conversion.utils import generate_dataset_json
import shutil
import gzip
from yucca.paths import yucca_raw_data
from tqdm import tqdm


def convert(path: str, subdir: str = "OASIS"):
    # INPUT DATA
    path = join(path, subdir)
    file_suffix = ".nii"

    # Train/Test Splits
    images_dir_tr = join(path, "Images", "Train")
    training_samples = subfiles(images_dir_tr, join=False, suffix=file_suffix)

    images_dir_ts = join(path, "Images", "Test")
    test_samples = subfiles(images_dir_ts, join=False, suffix=file_suffix)

    # OUTPUT DATA
    # Target names
    task_name = "Task701_OASIS_NoLabel"
    prefix = "OASIS"

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
        serial_number = sTr[:-4]
        image_file = open(join(images_dir_tr, sTr), "rb")
        shutil.copyfileobj(image_file, gzip.open(f"{target_imagesTr}/{prefix}_{serial_number}_000.nii.gz", "wb"))

    for sTs in tqdm(test_samples, desc="Test"):
        serial_number = sTs[:-4]
        image_file = open(join(images_dir_ts, sTs), "rb")
        shutil.copyfileobj(image_file, gzip.open(f"{target_imagesTs}/{prefix}_{serial_number}_000.nii.gz", "wb"))

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        ("T1",),
        labels={},
        dataset_name=task_name,
        license="hands off!",
        dataset_description="OASIS Dataset",
        dataset_reference="",
    )
