# %%
from yucca.training.managers.YuccaManager import YuccaManager


class YuccaManager_NoPatches(YuccaManager):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            patch_based_training=False,
            *args,
            **kwargs,
        )


# %%
