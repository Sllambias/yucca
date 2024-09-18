import lightning as pl
import torchvision
import logging
import torch
from typing import Literal, Optional, Union
from torch.utils.data import DataLoader, Sampler
from batchgenerators.utilities.file_and_folder_operations import join
from yucca.pipeline.configuration.split_data import SplitConfig
from yucca.modules.data.datasets.YuccaDataset import YuccaTestDataset, YuccaTrainDataset
from yucca.modules.data.samplers import InfiniteRandomSampler
from yucca.functional.array_operations.collate import single_case_collate


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
        batch_size: int,
        patch_size: tuple,
        allow_missing_modalities: Optional[bool] = False,
        image_extension: Optional[str] = None,
        composed_train_transforms: Optional[torchvision.transforms.Compose] = None,
        composed_val_transforms: Optional[torchvision.transforms.Compose] = None,
        num_workers: Optional[int] = None,
        overwrite_predictions: bool = False,
        pred_data_dir: Optional[str] = None,
        pred_include_cases: Optional[list] = None,
        pred_save_dir: Optional[str] = None,
        pre_aug_patch_size: Optional[Union[list, tuple]] = None,
        p_oversample_foreground: Optional[float] = 0.33,
        splits_config: Optional[SplitConfig] = None,
        split_idx: Optional[int] = None,
        task_type: Optional[str] = None,
        test_dataset_class: Optional[torch.utils.data.Dataset] = YuccaTestDataset,
        train_data_dir: Optional[str] = None,
        train_dataset_class: Optional[torch.utils.data.Dataset] = YuccaTrainDataset,
        train_sampler: Optional[Sampler] = InfiniteRandomSampler,
        val_sampler: Optional[Sampler] = InfiniteRandomSampler,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.image_extension = image_extension
        self.task_type = task_type

        self.split_idx = split_idx
        self.splits_config = splits_config
        self.train_data_dir = train_data_dir

        self.allow_missing_modalities = allow_missing_modalities
        self.composed_train_transforms = composed_train_transforms
        self.composed_val_transforms = composed_val_transforms
        self.pre_aug_patch_size = pre_aug_patch_size
        self.p_oversample_foreground = p_oversample_foreground

        # Set in the predict loop
        self.pred_include_cases = pred_include_cases
        self.overwrite_predictions = overwrite_predictions
        self.pred_data_dir = pred_data_dir
        self.pred_save_dir = pred_save_dir

        # Set default values
        self.num_workers = max(0, int(torch.get_num_threads() - 1)) if num_workers is None else num_workers
        self.val_num_workers = self.num_workers
        self.test_dataset_class = test_dataset_class
        self.train_sampler = train_sampler
        self.train_dataset_class = train_dataset_class
        self.val_sampler = val_sampler
        logging.info(f"Using {self.num_workers} workers")
        logging.info(
            f"Using dataset class: {self.train_dataset_class} for train/val and {self.test_dataset_class} for inference"
        )

    def setup(self, stage: Literal["fit", "test", "predict"]):
        logging.info(f"Setting up data for stage: {stage}")
        expected_stages = ["fit", "test", "predict"]
        assert stage in expected_stages, "unexpected stage. " f"Expected: {expected_stages} and found: {stage}"

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            assert self.train_data_dir is not None
            assert self.split_idx is not None
            assert self.splits_config is not None
            assert self.task_type is not None

            self.train_samples = [join(self.train_data_dir, i) for i in self.splits_config.train(self.split_idx)]
            self.val_samples = [join(self.train_data_dir, i) for i in self.splits_config.val(self.split_idx)]

            if len(self.train_samples) < 100:
                logging.info(f"Training on samples: {self.train_samples}")

            if len(self.val_samples) < 100:
                logging.info(f"Validating on samples: {self.val_samples}")

            self.train_dataset = self.train_dataset_class(
                self.train_samples,
                composed_transforms=self.composed_train_transforms,
                patch_size=self.pre_aug_patch_size if self.pre_aug_patch_size is not None else self.patch_size,
                task_type=self.task_type,
                allow_missing_modalities=self.allow_missing_modalities,
                p_oversample_foreground=self.p_oversample_foreground,
            )

            self.val_dataset = self.train_dataset_class(
                self.val_samples,
                composed_transforms=self.composed_val_transforms,
                patch_size=self.patch_size,
                task_type=self.task_type,
                allow_missing_modalities=self.allow_missing_modalities,
                p_oversample_foreground=self.p_oversample_foreground,
            )

        if stage == "predict":
            assert self.pred_data_dir is not None, "`pred_data_dir` is required in inference"
            assert self.pred_save_dir is not None, "`pred_save_dir` is required in inference"
            assert self.image_extension is not None, "`image_extension` is required in inference"
            # This dataset contains ONLY the images (and not the labels)
            # It will return a tuple of (case, case_id)
            self.pred_dataset = self.test_dataset_class(
                self.pred_data_dir,
                pred_save_dir=self.pred_save_dir,
                overwrite_predictions=self.overwrite_predictions,
                suffix=self.image_extension,
                pred_include_cases=self.pred_include_cases,
            )

    def train_dataloader(self):
        logging.info(f"Starting training with data from: {self.train_data_dir}")
        sampler = self.train_sampler(self.train_dataset) if self.train_sampler is not None else None
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=torch.cuda.is_available(),
            sampler=sampler,
            shuffle=sampler is None,
        )

    def val_dataloader(self):
        sampler = self.val_sampler(self.val_dataset) if self.val_sampler is not None else None
        return DataLoader(
            self.val_dataset,
            num_workers=self.val_num_workers,
            batch_size=self.batch_size,
            pin_memory=torch.cuda.is_available(),
            sampler=sampler,
        )

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        logging.info("Starting inference")
        return DataLoader(self.pred_dataset, num_workers=self.num_workers, batch_size=1, collate_fn=single_case_collate)
