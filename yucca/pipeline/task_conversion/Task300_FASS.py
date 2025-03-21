if __name__ == "__main__":
    import os
    import numpy as np
    from PIL import Image
    from batchgenerators.utilities.file_and_folder_operations import (
        maybe_mkdir_p,
        join,
        subfiles,
        save_json,
    )
    from yucca.pipeline.task_conversion.utils import generate_dataset_json
    from yucca.paths import get_raw_data_path, get_preprocessed_data_path

    """
    FASS nnU-Net task conversion script.
    """

    base = "/zhome/af/0/210164/data/FASS/task_converted"
    file_extension = ".png"
    target_dataset_id = 300
    target_dataset_name = f"Task{target_dataset_id:3.0f}_FASS"

    raw_dir = get_raw_data_path()
    maybe_mkdir_p(join(raw_dir, target_dataset_name))

    target_imagesTr = join(raw_dir, target_dataset_name, "imagesTr")
    target_labelsTr = join(raw_dir, target_dataset_name, "labelsTr")
    target_imagesTs = join(raw_dir, target_dataset_name, "imagesTs")
    target_labelsTs = join(raw_dir, target_dataset_name, "labelsTs")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTr)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTs)

    tr_cases = subfiles(os.path.join(base, "train", "images"), join=False)
    val_cases = subfiles(os.path.join(base, "val", "images"), join=False)
    test_cases = subfiles(os.path.join(base, "test", "images"), join=False)

    tr_split = []
    val_split = []

    for case in tr_cases:
        case = case[: -len(file_extension)]

        image = Image.open(join(base, "train", "images", case + file_extension))
        image = np.array(image)[:, :, 0]
        image = Image.fromarray(image)
        image.save(join(target_imagesTr, case + "_000.png"))

        label = Image.open(join(base, "train", "annotations", case + file_extension))
        label.save(join(target_labelsTr, case + ".png"))

        tr_split.append(case)

    for case in val_cases:
        case = case[: -len(file_extension)]

        image = Image.open(join(base, "val", "images", case + file_extension))
        image = np.array(image)[:, :, 0]
        image = Image.fromarray(image)
        image.save(join(target_imagesTr, case + "_000.png"))

        label = Image.open(join(base, "val", "annotations", case + file_extension))
        label.save(join(target_labelsTr, case + ".png"))

        val_split.append(case)

    for case in test_cases:
        case = case[: -len(file_extension)]

        image = Image.open(join(base, "test", "images", case + file_extension))
        image = np.array(image)[:, :, 0]
        image = Image.fromarray(image)
        image.save(join(target_imagesTs, case + "_000.png"))

        label = Image.open(join(base, "test", "annotations", case + file_extension))
        label.save(join(target_labelsTs, case + ".png"))

    # manual splits without extensions or modality encoding
    splits = [{"train": tr_split, "val": val_split}]

    pp_out_dir = join(get_preprocessed_data_path(), target_dataset_name)
    maybe_mkdir_p(pp_out_dir)
    save_json(splits, join(pp_out_dir, "splits_final.json"), sort_keys=False)

    labels = {
        0: "background",
        1: "artery",
        2: "liver",
        3: "stomach",
        4: "vein",
    }

    generate_dataset_json(
        join(raw_dir, target_dataset_name, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        ("Gray",),
        labels=labels,
        dataset_name=target_dataset_name,
    )
