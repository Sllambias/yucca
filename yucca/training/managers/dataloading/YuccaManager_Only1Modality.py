from yucca.training.managers.YuccaManager import YuccaManager
from yucca.training.data_loading.alternative_datasets.YuccaDataset_1modality import YuccaTrainDataset_1modality


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
        self.train_dataset = YuccaTrainDataset_1modality
