# %%
def test_manager_works():
    import os
    from yucca.training.trainers.YuccaConfigurator import YuccaConfigurator
    from yucca.paths import yucca_preprocessed

    print(yucca_preprocessed)
    print(os.path.isdir(yucca_preprocessed))
    print(os.listdir(yucca_preprocessed))
    print(os.path.isdir("/home"))
    print(os.path.isdir("/home/runner"))
    print(os.path.isdir("/home/runner/work"))
    print(os.path.isdir("/home/runner/work/yucca"))
    print(os.path.isdir("/home/runner/work/yucca/tests"))
    print(os.path.isdir("/home/runner/work/yucca/tests/data"))
    print(os.listdir("/home/runner/work/yucca/tests/data"))
    print(os.path.isdir("/home/runner/work/yucca/tests/data/Task000_Test"))
    print(os.path.isdir("/home/runner/work/yucca/tests/data/Task000_Test/YuccaPlanner"))
    print(os.getcwd())

    configurator = YuccaConfigurator(task="Task000_Test")
    assert configurator is not None


# %%
