from yucca.pipeline.managers.YuccaManagerV2 import YuccaManager


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
