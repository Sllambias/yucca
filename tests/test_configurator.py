def test_configurator_works():
    import os
    from yucca.training.trainers.YuccaConfigurator import YuccaConfigurator
    from yucca.paths import yucca_preprocessed

    configurator = YuccaConfigurator(task="Task000_Test")
    assert configurator is not None
