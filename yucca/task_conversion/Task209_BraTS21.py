from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data
import shutil
from tqdm import tqdm
from os.path import basename


def convert(_path: str):
    # This script assumes TC has already been done on supervised version of dataset with
    # name `Task012_BratS21`

    # Target names
    sup_task_name = "Task012_BraTS21"
    ssl_task_name = "Task209_BraTS21"

    source_folder = join(yucca_raw_data, sup_task_name, "imagesTr")
    target_base = join(yucca_raw_data, ssl_task_name)

    target_imagesTr = join(target_base, "imagesTr")
    maybe_mkdir_p(target_imagesTr)

    suffix = ".nii.gz"
    training_samples = subfiles(source_folder, suffix=suffix)

    ###Populate Target Directory###
    for src_path in tqdm(training_samples):
        file_name = basename(src_path)
        img_name = file_name[:-11]  # removing last _00x_.nii.gz

        if "_000." in file_name:
            dest_path = f"{target_imagesTr}/{img_name}_flair_000.nii.gz"
        elif "_001." in file_name:
            dest_path = f"{target_imagesTr}/{img_name}_t1_000.nii.gz"
        elif "_002." in file_name:
            dest_path = f"{target_imagesTr}/{img_name}_t1ce_000.nii.gz"
        elif "_003." in file_name:
            dest_path = f"{target_imagesTr}/{img_name}_t2_000.nii.gz"
        else:
            raise ValueError

        shutil.copy2(src_path, dest_path)

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        imagesTs_dir=None,
        modalities=("MRI"),
        labels=None,
        dataset_name=sup_task_name,
        license="hands off!",
        dataset_description="BraTS21",
        dataset_reference="https://www.nitrc.org/projects/msseg, https://arxiv.org/abs/2206.06694",
    )
