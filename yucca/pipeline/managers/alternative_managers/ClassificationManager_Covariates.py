from yucca.pipeline.managers.ClassificationManager import ClassificationManagerV2, ClassificationManagerV9
from yucca.modules.lightning_modules.ClassificationLightningModule_Covariates import ClassificationLightningModule_Covariates
from yucca.modules.data.datasets.ClassificationDataset import (
    ClassificationTrainDatasetWithCovariates,
)


class ClassificationManagerV9_Cov(ClassificationManagerV9):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "resnet18_2cov"
        self.lightning_module = ClassificationLightningModule_Covariates
        self.train_dataset_class = ClassificationTrainDatasetWithCovariates
        # self.test_dataset_class = ClassificationTestDatasetWithCovariates


class ClassificationManagerV9_DenseCov(ClassificationManagerV9):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "densenet121_2cov"
        self.lightning_module = ClassificationLightningModule_Covariates
        self.train_dataset_class = ClassificationTrainDatasetWithCovariates
        # self.test_dataset_class = ClassificationTestDatasetWithCovariates
