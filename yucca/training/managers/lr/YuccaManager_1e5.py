from yucca.training.managers.YuccaManager import YuccaManager


class YuccaManager_1e5(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = 1e-5
