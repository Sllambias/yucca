def test_manager_works():
    from yucca.yucca.training.trainers.YuccaManager import YuccaLightningManager

    manager = YuccaLightningManager()
    assert manager is not None
