import lightning as pl
import torchvision
from typing import Literal
from torch.utils.data import DataLoader, Sampler
from batchgenerators.utilities.file_and_folder_operations import join
from yucca.training.configuration.input_dimensions import InputDimensionsConfig
from yucca.training.configuration.split_data import SplitConfig
from yucca.training.configuration.configure_plans import PlanConfig
from yucca.training.data_loading.YuccaDataset import YuccaTestDataset, YuccaTrainDataset
from yucca.training.data_loading.samplers import InfiniteRandomSampler


class YuccaDataModule(pl.LightningDataModule):
    """
    The YuccaDataModule class is a PyTorch Lightning DataModule designed for handling data loading
    and preprocessing in the context of the Yucca project.

    It extends the pl.LightningDataModule class and provides methods for preparing data, setting up
    datasets for training, validation, and prediction, as well as creating data loaders for these stages.

    configurator (YuccaConfigurator): An instance of the YuccaConfigurator class containing configuration parameters.

    composed_train_transforms (torchvision.transforms.Compose, optional): A composition of transforms to be applied to the training dataset. Default is None.
    composed_val_transforms (torchvision.transforms.Compose, optional): A composition of transforms to be applied to the validation dataset. Default is None.

    num_workers (int, optional): Number of workers for data loading. Default is 8.

    pred_data_dir (str, optional): Directory containing data for prediction. Required only during the "predict" stage.

    pre_aug_patch_size (list or tuple, optional): Patch size before data augmentation. Default is None.
        - The purpose of the pre_aug_patch_size is to increase computational efficiency while not losing important information.
        If we have a volume of 512x512x512 and our model only works with patches of 128x128x128 there's no reason to
        apply the transform to the full volume. To avoid this we crop the volume before transforming it.

        But, we do not want to crop it to 128x128x128 before transforming it. Especially not before applying spatial transforms.
        Both because
        (1) the edges will contain a lot of border interpolation artifacts, and
        (2) if we crop to 128x128x128 and then rotate the image 45 degrees or downscale it (zoom out effect)
        we suddenly introduce dark areas where they should not be. We could've simply kept more of the original
        volume BEFORE scaling or rotating, then our 128x128x128 wouldn't be part-black.

        Therefore the pre_aug_patch_size parameter allows users to specify a patch size before augmentation is applied.
        This can potentially avoid dark or low-intensity areas at the borders and it also helps mitigate the risk of
        introducing artifacts during data augmentation, especially in regions where interpolation may have a significant impact.
    """

    def __init__(
        self,
        train_data_dir: str,
        input_dims_config: InputDimensionsConfig,
        plan_config: PlanConfig,
        splits_config: SplitConfig,
        split_idx: int,
        composed_train_transforms: torchvision.transforms.Compose = None,
        composed_val_transforms: torchvision.transforms.Compose = None,
        num_workers: int = 8,
        pred_data_dir: str = None,
        pre_aug_patch_size: list | tuple = None,
        sampler: Sampler = InfiniteRandomSampler,
    ):
        super().__init__()
        # extract parameters
        self.batch_size = input_dims_config.batch_size
        self.patch_size = input_dims_config.patch_size
        self.image_extension = plan_config.image_extension
        self.task_type = plan_config.task_type
        self.train_split = splits_config.train(split_idx)
        self.val_split = splits_config.val(split_idx)

        # Set by initialize()
        self.composed_train_transforms = composed_train_transforms
        self.composed_val_transforms = composed_val_transforms
        self.pre_aug_patch_size = pre_aug_patch_size

        # Set in the train loop
        self.train_data_dir = train_data_dir

        # Set in the predict loop
        self.pred_data_dir = pred_data_dir

        # Set default values
        self.num_workers = num_workers
        self.val_num_workers = num_workers // 2 if num_workers > 0 else num_workers
        self.sampler = sampler

    def prepare_data(self):
        self.train_samples = [join(self.train_data_dir, i) for i in self.train_split]
        self.val_samples = [join(self.train_data_dir, i) for i in self.val_split]

    def setup(self, stage: Literal["fit", "test", "predict"]):
        print(f"Setting up data for stage: {stage}")
        expected_stages = ["fit", "test", "predict"]
        assert stage in expected_stages, "unexpected stage. " f"Expected: {expected_stages} and found: {stage}"

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = YuccaTrainDataset(
                self.train_samples,
                composed_transforms=self.composed_train_transforms,
                patch_size=self.pre_aug_patch_size if self.pre_aug_patch_size is not None else self.patch_size,
                task_type=self.task_type,
            )

            self.val_dataset = YuccaTrainDataset(
                self.val_samples,
                composed_transforms=self.composed_val_transforms,
                patch_size=self.patch_size,
                task_type=self.task_type,
            )

        if stage == "predict":
            assert self.pred_data_dir is not None, "set a pred_data_dir for inference to work"
            # This dataset contains ONLY the images (and not the labels)
            # It will return a tuple of (case, case_id)
            self.pred_dataset = YuccaTestDataset(self.pred_data_dir, patch_size=self.patch_size, suffix=self.image_extension)

    def train_dataloader(self):
        print(f"Starting training with data from: {self.train_data_dir}")
        train_sampler = self.sampler(self.train_dataset) if self.sampler is not None else None
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            persistent_workers=bool(self.num_workers),
            sampler=train_sampler,
            shuffle=train_sampler is None,
        )

    def val_dataloader(self):
        val_sampler = self.sampler(self.val_dataset) if self.sampler is not None else None
        return DataLoader(
            self.val_dataset,
            num_workers=self.val_num_workers,
            batch_size=self.batch_size,
            persistent_workers=bool(self.val_num_workers),
            sampler=val_sampler,
        )

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        print("Starting inference")
        return DataLoader(self.pred_dataset, num_workers=self.num_workers, batch_size=1)
