# %%
def test_manager_works():
    import os
    from yucca.training.trainers.YuccaConfigurator import YuccaConfigurator
    from yucca.paths import yucca_preprocessed

    print(yucca_preprocessed)
    print(os.path.isdir(yucca_preprocessed))
    print(os.listdir(yucca_preprocessed))

    configurator = YuccaConfigurator(task="Task000_Test")
    assert configurator is not None


# %%
