from yucca.pipeline.managers.YuccaManager import YuccaManager
from yucca.modules.data.augmentation.augmentation_presets import generic
from yucca.modules.lightning_modules.YuccaLightningModule_skeleton_loss import YuccaLightningModule_skeleton_loss


class YuccaManager_SkeletonLoss(YuccaManager):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lightning_module = YuccaLightningModule_skeleton_loss
        self.loss = "DC_SkelREC_and_CE_loss"
        self.augmentation_params = generic
        self.augmentation_params["skeleton"] = True
        self.augmentation_params["do_tubes"] = True
