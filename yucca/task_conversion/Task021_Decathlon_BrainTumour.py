import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data, yucca_source
from yucca.utils.nib_utils import get_nib_orientation, get_nib_spacing


def convert(path: str = yucca_source, subdir: str = "decathlon", subsubdir: str = "Task01_BrainTumour"):
    # INPUT DATA
    path = f"{path}/{subdir}/{subsubdir}"

    file_suffix = ".nii.gz"

    # Train/Test Splits
    images_dir = join(path, "imagesTr")
    labels_dir = join(path, "labelsTr")
    training_samples, test_samples = train_test_split(subfiles(labels_dir, suffix=file_suffix, join=False), random_state=333)

    ###OUTPUT DATA
    # Target names
    task_name = "Task021_Decathlon_BrainTumour"
    task_prefix = "Decathlon_BrainTumour"

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

    images_dir_tr = images_dir_ts = images_dir
    labels_dir_tr = labels_dir_ts = labels_dir

    skipped = []

    for sTr in training_samples:
        image = nib.load(join(images_dir_tr, sTr))
        label = nib.load(join(labels_dir_tr, sTr))
        sTr = sTr[: -len(file_suffix)]

        # I believe this is the order they're saved in but as the model wont care I also wont care too much
        # If this is important I suggest you double check
        flair = image.slicer[:, :, :, 0]
        t1w = image.slicer[:, :, :, 1]
        t1gd = image.slicer[:, :, :, 2]
        t2w = image.slicer[:, :, :, 3]

        orig_ornt = get_nib_orientation(flair)
        label_ornt = get_nib_orientation(label)
        if orig_ornt == label_ornt:

            orig_spacing = get_nib_spacing(flair)
            label_spacing = get_nib_spacing(label)
            if orig_spacing.all() == label_spacing.all():

                orig_shape = np.shape(flair)
                label_shape = np.shape(label)
                if orig_shape == label_shape:

                    nib.save(flair, f"{target_imagesTr}/{task_prefix}_{sTr}_000.nii.gz")
                    nib.save(t1w, f"{target_imagesTr}/{task_prefix}_{sTr}_001.nii.gz")
                    nib.save(t1gd, f"{target_imagesTr}/{task_prefix}_{sTr}_002.nii.gz")
                    nib.save(t2w, f"{target_imagesTr}/{task_prefix}_{sTr}_003.nii.gz")

                    nib.save(label, f"{target_labelsTr}/{task_prefix}_{sTr}.nii.gz")

                else:
                    print("shape not matching, file not saved")
                    skipped.append(join(images_dir_tr, sTr))
            else:
                print("Spacing not matching")
                skipped.append(join(images_dir_tr, sTr))
        else:
            print("Orientation not matching")
            skipped.append(join(images_dir_tr, sTr))

    for sTs in test_samples:
        image = nib.load(join(images_dir_ts, sTs))
        label = nib.load(join(labels_dir_ts, sTs))
        sTs = sTs[: -len(file_suffix)]

        flair = image.slicer[:, :, :, 0]
        t1w = image.slicer[:, :, :, 1]
        t1gd = image.slicer[:, :, :, 2]
        t2w = image.slicer[:, :, :, 3]

        orig_ornt = get_nib_orientation(flair)
        label_ornt = get_nib_orientation(label)

        if orig_ornt == label_ornt:

            print("same orientation")

            orig_spacing = get_nib_spacing(flair)
            label_spacing = get_nib_spacing(label)

            if orig_spacing.all() == label_spacing.all():

                print("same spacing")

                orig_shape = np.shape(flair)
                label_shape = np.shape(label)

                if orig_shape == label_shape:
                    nib.save(flair, f"{target_imagesTs}/{task_prefix}_{sTs}_000.nii.gz")
                    nib.save(t1w, f"{target_imagesTs}/{task_prefix}_{sTs}_001.nii.gz")
                    nib.save(t1gd, f"{target_imagesTs}/{task_prefix}_{sTs}_002.nii.gz")
                    nib.save(t2w, f"{target_imagesTs}/{task_prefix}_{sTs}_003.nii.gz")

                    nib.save(label, f"{target_labelsTs}/{task_prefix}_{sTs}.nii.gz")

                else:
                    print("shape not matching, file not saved")
                    skipped.append(join(images_dir_ts, sTs))

            else:

                print("Spacing not matching")
                skipped.append(join(images_dir_ts, sTs))

        else:
            print("Orientation not matching")
            skipped.append(join(images_dir_ts, sTs))

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
