#%%
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, load_pickle
import lightning as pl
import random
import numpy as np
import torchvision
from torchvision import transforms
import numpy as np
from scipy.ndimage import map_coordinates
from yuccalib.image_processing.matrix_ops import create_zero_centered_coordinate_matrix, \
	deform_coordinate_matrix, Rx, Ry, Rz, Rz2D
from yuccalib.image_processing.transforms.YuccaTransform import YuccaTransform
from typing import Tuple
from yuccalib.image_processing.transforms.BiasField import BiasField
from yuccalib.image_processing.transforms.Spatial import Spatial
from yuccalib.image_processing.transforms.formatting import NumpyToTorch


class YuccaDataset(Dataset):
	def __init__(self, preprocessed_data_dir, composed_transforms: torchvision.transforms.Compose,
			  	 keep_in_ram: bool = False):
		self.data_path = preprocessed_data_dir
		self.files = np.array(subfiles(self.data_path, suffix='.npy', join=False))
		self.keep_in_ram = keep_in_ram
		self.transform = composed_transforms
		self.already_loaded_cases = []

	def load_and_maybe_keep_pickle(self, picklepath):
		if not self.keep_in_ram:
			return load_pickle(picklepath)
		if picklepath in self.already_loaded_cases:
			return self.already_loaded_cases[picklepath]
		self.already_loaded_cases[picklepath] = load_pickle(picklepath)
		return self.already_loaded_cases[picklepath]
	
	def load_and_maybe_keep_volume(self, path):
		if not self.keep_in_ram:
			if path[-3:] == 'npy':
				return np.load(path, 'r')
			image = np.load(path)
			assert len(image.files) == 1, "More than one entry in data array. "\
				f"Should only be ['data'] but is {[key for key in image.files]}"
			return image[image.files[0]]

		if path in self.already_loaded_cases:
			return self.already_loaded_cases[path]

		if path[-3:] == 'npy':
			try:
				self.already_loaded_cases[path] = np.load(path, 'r')
			except ValueError:
				self.already_loaded_cases[path] = np.load(path, allow_pickle=True)
			return self.already_loaded_cases[path]

		image = np.load(path)
		assert len(image.files) == 1, "More than one entry in data array. "\
			f"Should only be ['data'] but is {[key for key in image.files]}"
		self.already_loaded_cases = image[image.files[0]]
		return self.already_loaded_cases[path]

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		case = self.files[idx]
		#print(data.shape)
		data = np.load(join(self.data_path, case))#['data']
		data = {'image': data[:-1][:, :16, :16, :16], 'seg': data[-1:][:, :16, :16, :16]}
		return self.transform(data)

class InfiniteRandomBatchSampler(Sampler) :
	def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int = None):
		assert len(dataset) > 0
		self.dataset = dataset
		self.batch_size = batch_size
	
	def __iter__(self):
		yield np.random.choice(len(self.dataset), self.batch_size)
		

class YuccaDataModule(pl.LightningDataModule):
	def __init__(self, preprocessed_data_dir: str = "./", batch_size):
		super().__init__()
		self.batch_size = batch_size
		self.preprocessed_data_dir = preprocessed_data_dir
		self.files = subfiles(self.preprocessed_data_dir, suffix='.npy', join=False)
		self.composed_transforms = transforms.Compose([Spatial(patch_size=(16, 16, 16)),
												 BiasField(),
												 NumpyToTorch()])

	def prepare_data(self):
		pass

	def setup(self, stage: str):
		# Assign train/val datasets for use in dataloaders
		self.train_dataset = YuccaDataset(self.preprocessed_data_dir,
									composed_transforms=self.composed_transforms,
									keep_in_ram=True)

	def train_dataloader(self):
		train_sampler = InfiniteRandomBatchSampler(self.train_dataset, batch_size=self.batch_size)
		return DataLoader(self.train_dataset, num_workers=0, batch_sampler=train_sampler)

	def val_dataloader(self):
		return YuccaDataset(self.mnist_val, batch_size=32)

	def test_dataloader(self):
		return YuccaDataset(self.mnist_test, batch_size=32)

	def predict_dataloader(self):
		return YuccaDataset(self.mnist_predict, batch_size=32)


dm = YuccaDataModule(r'/Users/zcr545/Desktop/Projects/YuccaData/yucca_preprocessed/Task001_OASIS/YuccaPlanner')
dm.setup("1")
tdl = dm.train_dataloader()
i = 0
while i < 10:
	for out in tdl:
		x = out
		print(x['image'].shape)
		i += 1
		#print(out)

#if __name__ == '__main__':
#    main()
# %%
# %%

