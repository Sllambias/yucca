from yucca.pipeline.managers.YuccaManagerV2 import YuccaManager
from yucca.modules.lightning_modules.YuccaLightningModule_onehot_labels import YuccaLightningModule_onehot_labels


class YuccaManager_labelregions(YuccaManager):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.use_label_regions = True
