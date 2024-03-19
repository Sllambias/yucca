from yucca.training.managers.YuccaManager import YuccaManager
from yucca.training.augmentation.augmentation_presets import all_always


class YuccaManager_AllAlways(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = all_always
