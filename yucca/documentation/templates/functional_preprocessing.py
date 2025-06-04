if __name__ == "__main__":
    import re
    import os
    import numpy as np
    import torch
    from batchgenerators.utilities.file_and_folder_operations import (
        subfiles,
        join,
        save_pickle,
        maybe_mkdir_p as ensure_dir_exists,
    )
    from yucca.paths import get_raw_data_path, get_preprocessed_data_path
    from yucca.documentation.templates.template_config import config
    from yucca.functional.preprocessing import preprocess_case_for_training_with_label, preprocess_case_for_inference
    from yucca.functional.utils.loading import read_file_to_nifti_or_np

    raw_images_dir = join(get_raw_data_path(), config["task"], "imagesTr")
    raw_labels_dir = join(get_raw_data_path(), config["task"], "labelsTr")
    test_raw_images_dir = join(get_raw_data_path(), config["task"], "imagesTs")

    target_dir = join(get_preprocessed_data_path(), config["task"], config["config_name"])
    test_target_dir = join(get_preprocessed_data_path(), config["task"] + "_test", config["config_name"])

    ensure_dir_exists(target_dir)
    ensure_dir_exists(test_target_dir)

    # Preprocess the training data
    subjects = [file[: -len(config["extension"])] for file in subfiles(raw_labels_dir, join=False) if not file.startswith(".")]

    for sub in subjects:
        images = [
            image_path
            for image_path in subfiles(raw_images_dir)
            if re.search(re.escape(sub) + "_" + r"\d{3}" + ".", os.path.split(image_path)[-1])
        ]
        images = [read_file_to_nifti_or_np(image) for image in images]
        label = read_file_to_nifti_or_np(join(raw_labels_dir, sub + config["extension"]))
        images, label, image_props = preprocess_case_for_training_with_label(
            images=images,
            label=label,
            normalization_operation=[config["norm_op"] for _ in config["modalities"]],
            allow_missing_modalities=False,
            enable_cc_analysis=False,
            crop_to_nonzero=config["crop_to_nonzero"],
        )
        images = np.vstack((np.array(images), np.array(label)[np.newaxis]), dtype=np.float32)

        save_path = join(target_dir, sub)
        np.save(save_path + ".npy", images)
        save_pickle(image_props, save_path + ".pkl")

    # Preprocess the test data
    subjects = [
        file[: -len("_000" + config["extension"])]
        for file in subfiles(test_raw_images_dir, join=False)
        if not file.startswith(".")
    ]

    for sub in subjects:
        images = [
            image_path
            for image_path in subfiles(test_raw_images_dir)
            if re.search(re.escape(sub) + "_" + r"\d{3}" + ".", os.path.split(image_path)[-1])
        ]
        images = [read_file_to_nifti_or_np(image) for image in images]
        images, image_props = preprocess_case_for_inference(
            crop_to_nonzero=config["crop_to_nonzero"],
            images=images,
            intensities=None,
            normalization_scheme=[config["norm_op"] for _ in config["modalities"]],
            patch_size=config["patch_size"],
            target_size=config["target_size"],
            target_spacing=config["target_spacing"],
            target_orientation=config["target_coordinate_system"],
        )
        save_path = join(test_target_dir, sub)
        torch.save(images, save_path + ".pt")
        save_pickle(image_props, save_path + ".pkl")
