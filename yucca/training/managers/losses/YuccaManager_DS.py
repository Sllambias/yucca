from yucca.training.managers.YuccaManager import YuccaManager
from yucca.training.loss_and_optim.loss_functions.deep_supervision import DeepSupervisionLoss


class YuccaManager_DS(YuccaManager):
    """
    The difference from YuccaTrainerV2 --> YuccaTrainerV3 is:
    - Introduces Deep Supervision
    - Uses data augmentation scheme V2
    """

    def __init__(
        self,
        model,
        model_dimensions: str,
        task: str,
        folds: str | int,
        plan_id: str,
        starting_lr: float = None,
        loss_fn: str = None,
        momentum: float = None,
        continue_training: bool = False,
        checkpoint: str = None,
        finetune: bool = False,
        fast_training: bool = False,
    ):
        super().__init__(
            model,
            model_dimensions,
            task,
            folds,
            plan_id,
            starting_lr,
            loss_fn,
            momentum,
            continue_training,
            checkpoint,
            finetune,
            fast_training,
        )
        self.deep_supervision = True

    def comprehensive_eval(self, pred, seg):
        pred = pred[0]
        seg = seg[0]
        super().comprehensive_eval(pred, seg)

    def initialize_loss_optim_lr(self):
        super().initialize_loss_optim_lr()
        self.loss_fn = DeepSupervisionLoss(self.loss_fn)

    def predict_folder(self, input_folder, output_folder, not_strict=True, save_softmax=False, overwrite=False, do_tta=True):
        ds = self.deep_supervision
        self.network.deep_supervision = False
        super().predict_folder(input_folder, output_folder, not_strict, save_softmax, overwrite, do_tta)
        self.network.deep_supervision = ds
