from yucca.pipeline.managers.YuccaManager import YuccaManager
from yucca.data.augmentation.augmentation_presets import generic


class YuccaManagerV12(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deep_supervision = True
        self.augmentation_params = generic
        self.momentum = 0.99
