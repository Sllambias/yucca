from yucca.pipeline.managers.YuccaManager import YuccaManager
from yucca.modules.data.datasets.alternative_datasets.YuccaDataset_1modality import YuccaTrainDataset_1modality


class YuccaManager_Only1Modality(YuccaManager):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.train_dataset_class = YuccaTrainDataset_1modality

    def get_plan_config(self, *args, **kwargs):
        plan_config = super().get_plan_config(*args, **kwargs)
        plan_config.num_classes = 1
        plan_config.plans["num_modalities"] = 1
        return plan_config
