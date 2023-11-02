import lightning as pl
import random
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from batchgenerators.utilities.file_and_folder_operations import subfiles
from yucca.training.data_loading.YuccaDataset import YuccaDataset
from yucca.training.data_loading.samplers import InfiniteRandomBatchSampler


class YuccaDataModule(pl.LightningDataModule):
	def __init__(self, 
			  preprocessed_data_dir: str = None, batch_size: int = 2,
			  composed_tr_transforms: torchvision.transforms.Compose = None,
			  composed_val_transforms: torchvision.transforms.Compose = None,
			  generator_patch_size: list | tuple = None
			  ):
		super().__init__()
		self.batch_size = batch_size
		self.preprocessed_data_dir = preprocessed_data_dir
		self.files = subfiles(self.preprocessed_data_dir, suffix='.npy', join=False)
		self.composed_tr_transforms = composed_tr_transforms
		self.composed_val_transforms = composed_val_transforms
		self.generator_patch_size = generator_patch_size
		self.sampler = InfiniteRandomBatchSampler
		
	def prepare_data(self):
		pass

	def setup(self, stage: str = 'train'):
		expected_stages = ['train', 'test', 'inference']
		assert stage in expected_stages, "unexpected stage. "\
			f"Expected: {expected_stages} and found: {stage}"
		
		# Assign train/val datasets for use in dataloaders
		if stage == 'train':
			self.train_dataset = YuccaDataset(
				self.preprocessed_data_dir, 
				keep_in_ram=True,
				composed_transforms=self.composed_tr_transforms,
				generator_patch_size=self.generator_patch_size)
			self.val_dataset = YuccaDataset(
				self.preprocessed_data_dir,
				keep_in_ram=True,
				composed_transforms=self.composed_val_transforms,
				generator_patch_size=self.generator_patch_size)
		
	def train_dataloader(self):
		train_sampler = self.sampler(self.train_dataset, batch_size=self.batch_size)
		return DataLoader(self.train_dataset, num_workers=0, batch_sampler=train_sampler)

	def val_dataloader(self):
		val_sampler = self.sampler(self.val_dataset, batch_size=self.batch_size)
		return DataLoader(self.val_dataset, num_workers=0, batch_sampler=val_sampler)

	def test_dataloader(self):
		return YuccaDataset(self.mnist_test, batch_size=32)

	def predict_dataloader(self):
		return YuccaDataset(self.mnist_predict, batch_size=32)

