# %%
def test_manager_works():
    import os
    from yucca.training.trainers.YuccaConfigurator import YuccaConfigurator

    configurator = YuccaConfigurator(task="Task000_Test")
    assert configurator is not None


# %%
