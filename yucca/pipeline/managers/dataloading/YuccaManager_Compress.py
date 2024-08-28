from yucca.pipeline.managers.YuccaManager import YuccaManager
from yucca.modules.data.datasets.YuccaCompressedDataset import YuccaCompressedTrainDataset


class YuccaManager_Compress(YuccaManager):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.train_dataset_class = YuccaCompressedTrainDataset
