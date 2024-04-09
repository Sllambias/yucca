from yucca.training.managers.YuccaManager import YuccaManager
from yucca.training.augmentation.augmentation_presets import all_always


class YuccaManager_PreserveRange(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = all_always
        self.augmentation_params["clip_to_input_range"] = True
