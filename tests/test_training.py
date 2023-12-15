def test_2D_training():
    from yucca.yucca.training.trainers.YuccaManager import YuccaLightningManager

    manager = YuccaLightningManager(
        task="Task000_Test",
        fast_dev_run=True,
        disable_logging=True,
        model_name="TinyUNet",
        model_dimensions="2D",
        precision="bf16-mixed",
        accelerator="cpu",
    )
    manager.run_training()


def test_3D_training():
    from yucca.yucca.training.trainers.YuccaManager import YuccaLightningManager

    manager = YuccaLightningManager(
        task="Task000_Test",
        fast_dev_run=True,
        disable_logging=True,
        model_name="TinyUNet",
        model_dimensions="3D",
        precision="bf16-mixed",
        accelerator="cpu",
    )
    manager.run_training()
