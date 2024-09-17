if __name__ == "__main__":
    import lightning as L
    import os
    import torch
    from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p as ensure_dir_exists
    from yucca.paths import (
        get_models_path,
        get_results_path,
        get_preprocessed_data_path,
        get_raw_data_path,
    )
    from yucca.modules.callbacks.prediction_writer import WritePredictionFromLogits
    from yucca.modules.lightning_modules.YuccaLightningModule import YuccaLightningModule
    from yucca.modules.data.data_modules.YuccaDataModule import YuccaDataModule
    from yucca.modules.data.datasets.YuccaDataset import YuccaTestPreprocessedDataset
    from yucca.pipeline.evaluation.YuccaEvaluator import YuccaEvaluator
    from yucca.documentation.templates.template_config import config

    ckpt_path = os.path.join(
        get_models_path(),
        config["task"],
        config["model_name"] + "__" + config["dims"],
        "__" + config["plans_name"],
        "default",
        "kfold_5_fold_0",
        "version_0",
        "checkpoints",
        "last.ckpt",
    )

    gt_path = os.path.join(get_raw_data_path(), config["task"], "labelsTs")
    target_data_path = os.path.join(get_preprocessed_data_path(), config["task"] + "_test", "demo")

    save_path = os.path.join(
        get_results_path(),
        config["task"],
        config["task"],
        config["model_name"] + "__" + config["dims"],
        "__" + config["plans_name"],
        "kfold_5_fold_0",
        "version_0",
        "best",
    )
    ensure_dir_exists(save_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["hyper_parameters"]["config"]

    pred_writer = WritePredictionFromLogits(output_dir=save_path, save_softmax=False, write_interval="batch")

    model_module = YuccaLightningModule(
        config=config,
        deep_supervision=config.get("deep_supervision"),
        optimizer_kwargs={"learning_rate": config.get("learning_rate"), "momentum": config.get("momentum")},
        loss_fn=config.get("loss_fn"),
        disable_inference_preprocessing=True,
    )

    data_module = YuccaDataModule(
        batch_size=config["batch_size"],
        patch_size=config["patch_size"],
        pred_save_dir=save_path,
        pred_data_dir=target_data_path,
        overwrite_predictions=True,
        image_extension=".nii.gz",
        test_dataset_class=YuccaTestPreprocessedDataset,
    )

    trainer = L.Trainer(
        callbacks=pred_writer,
        precision="32",
        enable_progress_bar=True,
    )

    trainer.predict(
        model=model_module,
        dataloaders=data_module,
        ckpt_path=ckpt_path,
        return_predictions=False,
    )

    evaluator = YuccaEvaluator(
        labels=["0", "1", "2"],
        folder_with_ground_truth=gt_path,
        folder_with_predictions=save_path,
        use_wandb=False,
    )
    evaluator.run()
