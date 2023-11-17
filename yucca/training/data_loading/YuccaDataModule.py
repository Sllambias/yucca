import lightning as pl
import torchvision
from typing import Literal
from torch.utils.data import DataLoader
from batchgenerators.utilities.file_and_folder_operations import join
from yucca.training.data_loading.YuccaDataset import YuccaTestDataset, YuccaTrainDataset
from yucca.training.data_loading.samplers import InfiniteRandomSampler
from yucca.training.trainers.YuccaConfigurator import YuccaConfigurator


class YuccaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        configurator: YuccaConfigurator,
        composed_train_transforms: torchvision.transforms.Compose = None,
        composed_val_transforms: torchvision.transforms.Compose = None,
        num_workers: int = 8,
        pred_data_dir: str = None,
    ):
        super().__init__()
        # First set our configurator object
        self.cfg = configurator

        # Now extract parameters from the cfg
        self.batch_size = self.cfg.batch_size
        self.pre_aug_patch_size = self.cfg.pre_aug_patch_size
        self.patch_size = self.cfg.patch_size
        self.train_data_dir = self.cfg.train_data_dir
        self.train_split = self.cfg.train_split
        self.val_split = self.cfg.val_split

        # Set by initialize()
        self.composed_train_transforms = composed_train_transforms
        self.composed_val_transforms = composed_val_transforms

        # Set in the predict loop
        self.pred_data_dir = pred_data_dir

        # Set default values
        self.num_workers = num_workers
        self.val_num_workers = num_workers // 2 if num_workers > 0 else num_workers
        self.sampler = InfiniteRandomSampler

    def prepare_data(self):
        self.train_samples = [join(self.train_data_dir, i) for i in self.train_split]
        self.val_samples = [join(self.train_data_dir, i) for i in self.val_split]

    def setup(self, stage: Literal["fit", "test", "predict"]):
        expected_stages = ["fit", "test", "predict"]
        assert stage in expected_stages, "unexpected stage. " f"Expected: {expected_stages} and found: {stage}"

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = YuccaTrainDataset(
                self.train_samples,
                keep_in_ram=True,
                composed_transforms=self.composed_train_transforms,
                patch_size=self.pre_aug_patch_size if self.composed_train_transforms is not None else self.patch_size,
            )

            self.val_dataset = YuccaTrainDataset(
                self.val_samples,
                keep_in_ram=True,
                composed_transforms=self.composed_val_transforms,
                patch_size=self.patch_size,
            )

        if stage == "predict":
            assert self.pred_data_dir is not None, "set a pred_data_dir for inference to work"
            # This dataset contains ONLY the images (and not the labels)
            # It will return a tuple of (case, case_id)
            self.pred_dataset = YuccaTestDataset(self.pred_data_dir, patch_size=self.patch_size)

    def train_dataloader(self):
        train_sampler = self.sampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            persistent_workers=bool(self.num_workers),
            sampler=train_sampler,
        )

    def val_dataloader(self):
        val_sampler = self.sampler(self.val_dataset)
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
        return DataLoader(self.pred_dataset, num_workers=self.num_workers, batch_size=1)
