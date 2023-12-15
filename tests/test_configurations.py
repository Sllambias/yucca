def test_configurator_works():
    import os
    from yucca.training.configuration.configure_callbacks import get_callback_config
    from yucca.training.configuration.configure_paths_and_plans import get_path_and_plan_config
    from yucca.training.configuration.input_dimensions import get_input_dims
    from yucca.training.configuration.split_data import get_split_config
    from yucca.paths import yucca_preprocessed_data

    path_and_plan_config = get_path_and_plan_config(task="Task000_Test", model_dimensions="2D", model_name="TinyUNet")
    input_dims = get_input_dims(
        plan=path_and_plan_config.plans,
        model_dimensions=path_and_plan_config.model_dimensions,
        num_classes=path_and_plan_config.num_classes,
        model_name=path_and_plan_config.model_name,
        batch_size="tiny",
        patch_size="tiny",
    )
    splits = get_split_config(train_data_dir=path_and_plan_config.train_data_dir, task=path_and_plan_config.task)
    callbacks = get_callback_config(
        task=path_and_plan_config.task,
        save_dir=path_and_plan_config.save_dir,
        version_dir=path_and_plan_config.version_dir,
        version=path_and_plan_config.version,
    )
