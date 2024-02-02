"""
ONLY used to run tests
The data in the test folder was generated using the following lines of code:

ims = []
labs = []
for i in range(7):
    arr = np.random.rand(32, 32, 32)
    lab = np.random.randint(0, 2, (32, 32, 32)).astype(float)
    ims.append(nib.Nifti1Image(arr, affine=np.eye(4)))
    labs.append(nib.Nifti1Image(lab, affine=np.eye(4)))

"""

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.task_conversion.utils import generate_dataset_json
import shutil
from yucca.paths import yucca_raw_data
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def convert(path: str, subdir: str = "dataset_test0"):
    # INPUT DATA
    path = join(path, subdir)
    suffix = ".nii.gz"

    # Train/Test Splits
    labels_dir = join(path, "all_labels")
    images_dir = join(path, "all_images")

    training_samples, test_samples = train_test_split(subfiles(labels_dir, join=False, suffix=suffix), random_state=859032)

    # OUTPUT DATA
    # Target names
    task_name = "Task000_TEST_SEGMENTATION"
    task_prefix = "TEST_SEGMENTATION"

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

    for sTr in tqdm(training_samples, desc="Train"):
        sTr = sTr[: -len(suffix)]
        src_image_file_path = join(images_dir, sTr + suffix)
        src_label_path = join(labels_dir, sTr + suffix)

        dst_image_file_path = f"{target_imagesTr}/{task_prefix}_{sTr}_000.nii.gz"
        dst_label_path = f"{target_labelsTr}/{task_prefix}_{sTr}.nii.gz"
        shutil.copy2(src_image_file_path, dst_image_file_path)
        shutil.copy2(src_label_path, dst_label_path)
    del sTr

    for sTs in tqdm(test_samples, desc="Test"):
        sTs = sTs[: -len(suffix)]

        # Loading relevant modalities and the ground truth
        src_image_file_path = join(images_dir, sTs + suffix)
        src_label_path = join(labels_dir, sTs + suffix)

        dst_image_file_path = f"{target_imagesTs}/{task_prefix}_{sTs}_000.nii.gz"
        dst_label_path = f"{target_labelsTs}/{task_prefix}_{sTs}.nii.gz"
        shutil.copy2(src_image_file_path, dst_image_file_path)
        shutil.copy2(src_label_path, dst_label_path)
    del sTs

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        ("TestModality",),
        labels={0: "background", 1: "FakeLabel1"},
        dataset_name=task_name,
        dataset_description="Test Dataset",
        dataset_reference="",
    )


# %%
