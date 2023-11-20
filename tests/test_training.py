def test_manager_works():
    from yucca.training.trainers.YuccaLightningManager import YuccaLightningManager

    manager = YuccaLightningManager(task="Task000_Test", fast_dev_run=True, disable_logging=True, model_name="TinyUNet")
    manager.run_training()
