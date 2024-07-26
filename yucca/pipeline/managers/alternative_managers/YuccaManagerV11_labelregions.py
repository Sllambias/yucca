from yucca.pipeline.managers.YuccaManagerV11 import YuccaManagerV11
from yucca.lightning_modules.YuccaLightningModule_onehot_labels import YuccaLightningModule_onehot_labels


class YuccaManagerV11_labelregions(YuccaManagerV11):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.lightning_module = YuccaLightningModule_onehot_labels
        self.use_label_regions = True
