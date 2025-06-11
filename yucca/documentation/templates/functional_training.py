if __name__ == "__main__":
    import lightning as L
    from yucca.pipeline.configuration.configure_task import TaskConfig
    from yucca.pipeline.configuration.configure_paths import get_path_config
    from yucca.pipeline.configuration.configure_callbacks import get_callback_config
    from yucca.pipeline.configuration.split_data import get_split_config
    from yucca.modules.data.augmentation.YuccaAugmentationComposer import YuccaAugmentationComposer
    from yucca.modules.data.augmentation.augmentation_presets import no_aug
    from yucca.modules.lightning_modules.BaseLightningModule import BaseLightningModule
    from yucca.modules.data.data_modules.YuccaDataModule import YuccaDataModule
    from yucca.documentation.templates.template_config import config

    task_config = TaskConfig(
        task=config.get("task"),
        continue_from_most_recent=True,
        experiment=config.get("experiment"),
        manager_name="",
        model_dimensions=config.get("model_dimensions"),
        model_name=config.get("model_name"),
        patch_based_training=True,
        planner_name=config.get("config_name"),
        split_idx=config.get("split_idx"),
        split_method=config.get("split_method"),
        split_param=config.get("split_param"),
    )

    path_config = get_path_config(task_config=task_config, stage="fit")

    split_config = get_split_config(method=task_config.split_method, param=task_config.split_param, path_config=path_config)

    callback_config = get_callback_config(
        save_dir=path_config.save_dir,
        version_dir=path_config.version_dir,
        experiment=task_config.experiment,
        version=path_config.version,
        latest_ckpt_epochs=1,
        enable_logging=False,
        store_best_ckpt=False,
    )

    augmenter = YuccaAugmentationComposer(
        deep_supervision=config.get("deep_supervision"),
        patch_size=config.get("patch_size"),
        is_2D=True if config.get("model_dimensions") == "2D" else False,
        parameter_dict=no_aug,
        task_type_preset=config.get("task_type"),
    )

    model_module = BaseLightningModule(
        model=config["model"],
        model_dimensions=config["model_dimensions"],
        num_classes=config["num_classes"],
        num_modalities=config["num_modalities"],
        patch_size=config["patch_size"],
        crop_to_nonzero=config["crop_to_nonzero"],
        deep_supervision=config["deep_supervision"],
        optimizer_kwargs={"lr": config.get("learning_rate"), "momentum": config.get("momentum")},
        loss_fn=config.get("loss_fn"),
    )

    data_module = YuccaDataModule(
        batch_size=config.get("batch_size"),
        patch_size=config.get("patch_size"),
        composed_train_transforms=augmenter.train_transforms,
        composed_val_transforms=augmenter.val_transforms,
        train_data_dir=path_config.train_data_dir,
        split_idx=task_config.split_idx,
        splits_config=split_config,
        task_type=config.get("task_type"),
    )

    trainer = L.Trainer(
        callbacks=callback_config.callbacks,
        default_root_dir=path_config.save_dir,
        limit_train_batches=2,
        limit_val_batches=2,
        logger=callback_config.loggers,
        precision="16",
        enable_progress_bar=True,
        max_epochs=config["max_epochs"],
        accelerator="cpu",
    )

    trainer.fit(
        model=model_module,
        datamodule=data_module,
        ckpt_path="last",
    )
