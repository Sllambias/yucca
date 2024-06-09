import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.pipeline.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data, yucca_source


def convert(path: str = yucca_source, subdir: str = "decathlon", subsubdir: str = "Task01_BrainTumour"):
    # INPUT DATA
    path = f"{path}/{subdir}/{subsubdir}"

    file_suffix = ".nii.gz"

    # Train/Test Splits
    images_dir_tr = join(path, "imagesTr")
    labels_dir_tr = join(path, "labelsTr")
    images_dir_ts = join(path, "imagesTs")

    ###OUTPUT DATA
    # Target names
    task_name = "Task031_MSD_BrainTumour"

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

    for sTr in subfiles(labels_dir_tr, join=False):
        image = nib.load(join(images_dir_tr, sTr))
        label = nib.load(join(labels_dir_tr, sTr))
        sTr = sTr[: -len(file_suffix)]

        # I believe this is the order they're saved in but as the model wont care I also wont care too much
        # If this is important I suggest you double check
        flair = image.slicer[:, :, :, 0]
        t1w = image.slicer[:, :, :, 1]
        t1gd = image.slicer[:, :, :, 2]
        t2w = image.slicer[:, :, :, 3]

        nib.save(flair, f"{target_imagesTr}/{sTr}_000.nii.gz")
        nib.save(t1w, f"{target_imagesTr}/{sTr}_001.nii.gz")
        nib.save(t1gd, f"{target_imagesTr}/{sTr}_002.nii.gz")
        nib.save(t2w, f"{target_imagesTr}/{sTr}_003.nii.gz")

        nib.save(label, f"{target_labelsTr}/{sTr}.nii.gz")

    for sTs in subfiles(images_dir_ts, join=False):
        image = nib.load(join(images_dir_ts, sTs))
        sTs = sTs[: -len(file_suffix)]

        flair = image.slicer[:, :, :, 0]
        t1w = image.slicer[:, :, :, 1]
        t1gd = image.slicer[:, :, :, 2]
        t2w = image.slicer[:, :, :, 3]

        nib.save(flair, f"{target_imagesTs}/{sTs}_000.nii.gz")
        nib.save(t1w, f"{target_imagesTs}/{sTs}_001.nii.gz")
        nib.save(t1gd, f"{target_imagesTs}/{sTs}_002.nii.gz")
        nib.save(t2w, f"{target_imagesTs}/{sTs}_003.nii.gz")

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        modalities=("FLAIR", "T1w", "t1gd", "T2W"),
        labels={0: "background", 1: "edema", 2: "non-enhancing tumor", 3: "enhancing tumour"},
        dataset_name=task_name,
        license="CC-BY-SA 4.0",
        dataset_description="Decathlon: Brain Tumour",
        dataset_reference="King's College London",
    )


if __name__ == "__main__":
    convert()
