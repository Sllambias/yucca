import lightning as pl
import random
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from batchgenerators.utilities.file_and_folder_operations import subfiles
from yucca.training.data_loading.YuccaDataset import YuccaTrainDataset, YuccaTestDataset
from yucca.training.data_loading.samplers import InfiniteRandomBatchSampler


class YuccaDataModule(pl.LightningDataModule):
	def __init__(self, 
			  train_data_dir: str = None, 
			  pred_data_dir: str = None,
			  batch_size: int = 2,
			  composed_train_transforms: torchvision.transforms.Compose = None,
			  composed_val_transforms: torchvision.transforms.Compose = None,
			  generator_patch_size: list | tuple = None
			  ):
		super().__init__()
		self.batch_size = batch_size
		self.train_data_dir = train_data_dir
		self.pred_data_dir = pred_data_dir
		self.composed_train_transforms = composed_train_transforms
		self.composed_val_transforms = composed_val_transforms
		self.generator_patch_size = generator_patch_size
		self.sampler = InfiniteRandomBatchSampler
		
	def prepare_data(self):
		pass

	def setup(self, stage: str = 'fit'):
		expected_stages = ['fit', 'test', 'predict']
		assert stage in expected_stages, "unexpected stage. "\
			f"Expected: {expected_stages} and found: {stage}"
		
		# Assign train/val datasets for use in dataloaders
		if stage == 'fit':
			self.train_dataset = YuccaTrainDataset(
				self.train_data_dir, 
				keep_in_ram=True,
				composed_transforms=self.composed_train_transforms,
				generator_patch_size=self.generator_patch_size)
			
			self.val_dataset = YuccaTrainDataset(
				self.train_data_dir,
				keep_in_ram=True,
				composed_transforms=self.composed_val_transforms,
				generator_patch_size=self.generator_patch_size)
			
		if stage == 'predict':
			self.pred_dataset = YuccaTestDataset(self.pred_data_dir)
		
	def train_dataloader(self):
		train_sampler = self.sampler(self.train_dataset, batch_size=self.batch_size)
		return DataLoader(self.train_dataset, num_workers=0, batch_sampler=train_sampler)

	def val_dataloader(self):
		val_sampler = self.sampler(self.val_dataset, batch_size=self.batch_size)
		return DataLoader(self.val_dataset, num_workers=0, batch_sampler=val_sampler)

	def test_dataloader(self):
		return None

	def predict_dataloader(self):
		return DataLoader(self.pred_dataset, batch_size=1)

