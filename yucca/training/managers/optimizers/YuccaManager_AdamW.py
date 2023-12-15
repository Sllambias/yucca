from yucca.training.managers.YuccaManager import YuccaManager
from torch import optim


class YuccaManager_AdamW(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim = optim.AdamW
