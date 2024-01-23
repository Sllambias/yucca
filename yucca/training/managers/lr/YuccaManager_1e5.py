from yucca.training.managers.YuccaManager import YuccaManager


class YuccaManager_1e5(YuccaManager):
    def __init__(self, learning_rate=1e-5, *args, **kwargs):
        super().__init__(learning_rate=learning_rate, *args, **kwargs)
