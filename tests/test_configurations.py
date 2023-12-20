def test_configurations():
    import os
    from yucca.training.configuration.configure_callbacks import get_callback_config
    from yucca.training.configuration.configure_paths import get_path_config
    from yucca.training.configuration.configure_plans import get_plan_config
    from yucca.training.configuration.configure_task import get_task_config
    from yucca.training.configuration.input_dimensions import get_input_dims_config
    from yucca.training.configuration.split_data import get_split_config
    from yucca.paths import yucca_preprocessed_data

    task_config = get_task_config(task="Task000_Test")
    assert task_config is not None and isinstance(task_config.continue_from_most_recent, bool)

    path_config = get_path_config(ckpt_path=None, task_config=task_config)
    assert path_config is not None and isinstance(path_config.version, int)

    plan_config = get_plan_config(path_config=path_config, continue_from_most_recent=True)
    assert plan_config is not None and plan_config.task_type in ["classification", "segmentation", "unsupervised"]

    input_dims = get_input_dims_config(
        plan=plan_config.plans,
        model_dimensions=task_config.model_dimensions,
        num_classes=plan_config.num_classes,
        model_name=task_config.model_name,
        batch_size="tiny",
        patch_size="tiny",
    )
    assert input_dims is not None and len(input_dims.patch_size) in [2, 3]

    split_config = get_split_config(train_data_dir=path_config.train_data_dir, task=task_config.task)
    assert split_config is not None and len(split_config.train(0)) > 0

    callback_config = get_callback_config(
        task=task_config.task,
        save_dir=path_config.save_dir,
        version_dir=path_config.version_dir,
        version=path_config.version,
    )
    assert callback_config is not None and isinstance(callback_config.loggers, list)
