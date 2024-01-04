def test_manager_works():
    from yucca.training.managers.YuccaManager import YuccaManager

    manager = YuccaManager(
        enable_logging=False,
    )
    assert manager is not None
