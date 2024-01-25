from yucca.training.managers.YuccaManager import YuccaManager


class YuccaManager_PatchSizeMax(YuccaManager):
    """
    Instantiate a YuccaLightningManager object with the patch size set to the maximum patch size
    (no sliding window prediction).
    """

    def __init__(
        self,
        ckpt_path: str = None,
        continue_from_most_recent: bool = True,
        deep_supervision: bool = False,
        loss: str = "DiceCE",
        max_epochs: int = 1000,
        model_dimensions: str = "3D",
        model_name: str = "TinyUNet",
        num_workers: int = 8,
        planner: str = "YuccaPlanner",
        precision: str = "16-mixed",
        profile: bool = False,
        step_logging: bool = False,
        task: str = None,
        **kwargs,
    ):
        super().__init__(
            ckpt_path=ckpt_path,
            continue_from_most_recent=continue_from_most_recent,
            deep_supervision=deep_supervision,
            loss=loss,
            max_epochs=max_epochs,
            model_dimensions=model_dimensions,
            model_name=model_name,
            num_workers=num_workers,
            planner=planner,
            precision=precision,
            profile=profile,
            step_logging=step_logging,
            task=task,
            patch_size="max",
            patch_based_training=False,
            **kwargs,
        )
