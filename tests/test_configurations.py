def test_configurations(setup_preprocessed_segmentation_data):
    from yucca.training.configuration.configure_callbacks import get_callback_config
    from yucca.training.configuration.configure_paths import get_path_config
    from yucca.training.configuration.configure_plans import get_plan_config
    from yucca.training.configuration.configure_task import get_task_config
    from yucca.training.configuration.configure_input_dims import get_input_dims_config
    from yucca.training.configuration.split_data import get_split_config
    from yucca.training.configuration.configure_checkpoint import get_checkpoint_config
    from yucca.training.configuration.configure_seed import seed_everything_and_get_seed_config

    task_config = get_task_config(task="Task000_TEST_SEGMENTATION")
    assert task_config is not None and isinstance(task_config.continue_from_most_recent, bool)

    path_config = get_path_config(task_config=task_config)
    assert path_config is not None and isinstance(path_config.version, int)

    ckpt_config = get_checkpoint_config(
        path_config=path_config,
        continue_from_most_recent=task_config.continue_from_most_recent,
        ckpt_path=None,
        current_experiment=task_config.experiment,
    )
    assert ckpt_config.ckpt_wandb_id is None or isinstance(ckpt_config.ckpt_wandb_id, str)

    seed_config = seed_everything_and_get_seed_config(ckpt_seed=ckpt_config.ckpt_seed)
    assert isinstance(seed_config.seed, int)

    plan_config = get_plan_config(
        ckpt_plans=ckpt_config.ckpt_plans,
        plans_path=path_config.plans_path,
        stage="fit",
    )
    assert plan_config is not None and plan_config.task_type in ["classification", "segmentation", "unsupervised"]

    splits_config = get_split_config(task_config.split_method, task_config.split_param, path_config)
    assert splits_config is not None and len(splits_config.train(0)) > 0

    input_dims = get_input_dims_config(
        plan=plan_config.plans,
        model_dimensions=task_config.model_dimensions,
        num_classes=plan_config.num_classes,
        model_name=task_config.model_name,
        batch_size="tiny",
        patch_size="tiny",
    )
    assert input_dims is not None and len(input_dims.patch_size) in [2, 3]

    callback_config = get_callback_config(
        enable_logging=False,
        save_dir=path_config.save_dir,
        version_dir=path_config.version_dir,
        version=path_config.version,
    )
    assert callback_config is not None and isinstance(callback_config.loggers, list)
