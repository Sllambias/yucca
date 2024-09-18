from yucca.pipeline.managers.YuccaManagerV2 import YuccaManagerV2
from yucca.modules.lightning_modules.YuccaLightningModule_onehot_labels import YuccaLightningModule_onehot_labels


class YuccaManagerV2_labelregions(YuccaManagerV2):
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
