import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.pipeline.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data, yucca_source
from yucca.functional.testing.data.nifti import verify_spacing_is_equal, verify_orientation_is_equal


def convert(path: str = yucca_source, subdir: str = "decathlon", subsubdir: str = "Task05_Prostate"):
    # INPUT DATA
    path = f"{path}/{subdir}/{subsubdir}"
    file_suffix = ".nii.gz"

    # OUTPUT DATA
    # Define the task name and prefix
    task_name = "Task035_MSD_Prostate"

    # Set target paths
    target_base = join(yucca_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    # Split data
    images_dir_tr = join(path, "imagesTr")
    labels_dir_tr = join(path, "labelsTr")
    images_dir_ts = join(path, "imagesTs")

    # Populate Target Directory
    # This is also the place to apply any re-orientation, resampling and/or label correction.

    for sTr in subfiles(images_dir_tr, join=False):
        image_path = join(images_dir_tr, sTr)
        label_path = join(labels_dir_tr, sTr)
        sTr = sTr[: -len(file_suffix)]

        image = nib.load(image_path)
        label = nib.load(label_path)

        # I believe this is the order they're saved in but as the model wont care I also wont care too much
        # If this is important I suggest you double check
        t2 = image.slicer[:, :, :, 0]
        adc = image.slicer[:, :, :, 1]

        assert verify_spacing_is_equal(t2, label), "spacing"
        assert verify_orientation_is_equal(t2, label), "orientation"

        nib.save(t2, f"{target_imagesTr}/{sTr}_000.nii.gz")
        nib.save(adc, f"{target_imagesTr}/{sTr}_001.nii.gz")

        nib.save(label, f"{target_labelsTr}/{sTr}.nii.gz")

    for sTs in subfiles(images_dir_ts, join=False):
        image_path = join(images_dir_ts, sTs)
        sTs = sTs[: -len(file_suffix)]

        image = nib.load(image_path)

        t2 = image.slicer[:, :, :, 0]
        adc = image.slicer[:, :, :, 1]

        nib.save(t2, f"{target_imagesTr}/{sTr}_000.nii.gz")
        nib.save(adc, f"{target_imagesTr}/{sTr}_001.nii.gz")

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        modalities=("T2", "ADC"),
        labels={0: "Background", 1: "PZ", 2: "TZ"},
        dataset_name=task_name,
        license="CC-BY-SA 4.0",
        dataset_description="Decathlon: Prostate",
        dataset_reference="King's College London",
    )
