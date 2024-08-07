from yucca.pipeline.managers.YuccaManager import YuccaManager


class YuccaManager_CE(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = "CE"
