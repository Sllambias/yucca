from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.task_conversion.utils import generate_dataset_json
import shutil
import gzip
from yucca.paths import yucca_raw_data
from multiprocessing import Pool


def convert_case(sample, input_image_dir, input_label_dir, target_image_dir, target_label_dir, prefix, suffix):
    serial_number = sample[: -len(suffix)]
    image_file = open(join(input_image_dir, sample), "rb")
    label = open(join(input_label_dir, sample), "rb")
    shutil.copyfileobj(image_file, gzip.open(f"{target_image_dir}/{prefix}_{serial_number}_000.nii.gz", "wb"))
    shutil.copyfileobj(label, gzip.open(f"{target_label_dir}/{prefix}_{serial_number}.nii.gz", "wb"))


def convert(path: str, subdir: str = "DatasetName"):
    # INPUT DATA
    path = join(path, subdir)
    file_suffix = ".nii"

    # Train/Test Splits
    labels_dir_tr = join(path, "Labels", "Train")
    images_dir_tr = join(path, "Images", "Train")
    training_samples = subfiles(labels_dir_tr, join=False, suffix=file_suffix)

    labels_dir_ts = join(path, "Labels", "Test")
    images_dir_ts = join(path, "Images", "Test")
    test_samples = subfiles(labels_dir_ts, join=False, suffix=file_suffix)

    # OUTPUT DATA
    # Target names
    task_name = "Task000_MyTask"
    prefix = "MyTask"

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

    tr_cases = [
        (sample, images_dir_tr, labels_dir_tr, target_imagesTr, target_labelsTr, prefix, file_suffix)
        for sample in training_samples
    ]

    ts_cases = [
        (sample, images_dir_ts, labels_dir_ts, target_imagesTs, target_labelsTs, prefix, file_suffix)
        for sample in test_samples
    ]

    all_cases = tr_cases + ts_cases

    p = Pool(8)
    p.starmap(convert_case, all_cases)
    p.close()
    p.join()

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        ("T1.. or maybe CT?",),
        labels={0: "background", 1: "Fake Label", 2: "Also Fake Label"},
        dataset_name=task_name,
        license="hands off!",
        dataset_description="Fake Dataset",
        dataset_reference="",
    )
