from yucca.managers.YuccaManager import YuccaManager
from yucca.augmentation.augmentation_presets import CT


class YuccaManager_CT(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = CT
