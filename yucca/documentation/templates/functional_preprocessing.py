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
        save_json,
    )
    from yucca.paths import get_raw_data_path, get_preprocessed_data_path
    from yucca.documentation.templates.template_config import config
    from yucca.functional.preprocessing import preprocess_case_for_training_with_label, preprocess_case_for_inference
    from yucca.functional.utils.loading import read_file_to_nifti_or_np
    from yucca.functional.planning import make_plans_file, add_stats_to_plans_post_preprocessing

    raw_images_dir = join(get_raw_data_path(), config["task"], "imagesTr")
    raw_labels_dir = join(get_raw_data_path(), config["task"], "labelsTr")
    test_raw_images_dir = join(get_raw_data_path(), config["task"], "imagesTs")

    target_dir = join(get_preprocessed_data_path(), config["task"], config["plans_name"])
    test_target_dir = join(get_preprocessed_data_path(), config["task"] + "_test", config["plans_name"])

    ensure_dir_exists(target_dir)
    ensure_dir_exists(test_target_dir)

    plans = make_plans_file(
        allow_missing_modalities=False,
        crop_to_nonzero=config["crop_to_nonzero"],
        norm_op=config["norm_op"],
        classes=[0, 1, 2],
        plans_name=config["plans_name"],
        modalities=config["modalities"],
        task_type=config["task_type"],
    )

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
            normalization_operation=plans["normalization_scheme"],
            allow_missing_modalities=False,
            enable_cc_analysis=False,
            crop_to_nonzero=plans["crop_to_nonzero"],
        )
        images = np.vstack((np.array(images), np.array(label)[np.newaxis]), dtype=np.float32)

        save_path = join(target_dir, sub)
        np.save(save_path + ".npy", images)
        save_pickle(image_props, save_path + ".pkl")

    plans = add_stats_to_plans_post_preprocessing(plans=plans, directory=target_dir)
    save_json(plans, join(target_dir, config["plans_name"] + "_plans.json"), sort_keys=False)

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
            crop_to_nonzero=plans["crop_to_nonzero"],
            keep_aspect_ratio=plans["keep_aspect_ratio_when_using_target_size"],
            images=images,
            intensities=None,
            normalization_scheme=["volume_wise_znorm"],
            patch_size=(32, 32),
            target_size=plans["target_size"],
            target_spacing=plans["target_spacing"],
            target_orientation=plans["target_coordinate_system"],
            transpose_forward=plans["transpose_forward"],
        )
        save_path = join(test_target_dir, sub)
        torch.save(images, save_path + ".pt")
        save_pickle(image_props, save_path + ".pkl")

    save_json(plans, join(test_target_dir, config["plans_name"] + "_plans.json"), sort_keys=False)
