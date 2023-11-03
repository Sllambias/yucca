import numpy as np
import torchvision
import nibabel as nib
import os
from batchgenerators.utilities.file_and_folder_operations import subfiles, load_pickle, join
from torch.utils.data import Dataset
from yuccalib.image_processing.transforms.cropping_and_padding import CropPad
from yucca.paths import yucca_raw_data

class YuccaTrainDataset(Dataset):
	def __init__(self, preprocessed_data_dir,
			  	 keep_in_ram: bool = False,
				 patch_size: list | tuple = None,
				 composed_transforms: torchvision.transforms.Compose = None):
		self.data_path = preprocessed_data_dir
		self.all_cases = np.array(subfiles(self.data_path, suffix='.npy', join=False))
		self.keep_in_ram = keep_in_ram
		self.croppad = CropPad(patch_size=patch_size, p_oversample_foreground=0.33)
		self.composed_transforms = composed_transforms
		self.already_loaded_cases = {}

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
		return len(self.all_cases)

	def __getitem__(self, idx):
		case = self.all_cases[idx]
		data = self.load_and_maybe_keep_volume(join(self.data_path, case))
		metadata = self.load_and_maybe_keep_pickle(join(self.data_path, case[:-len('.npy')] + '.pkl'))
		data_dict = {'image': data[:-1], 'seg': data[-1:]}
		data_dict = self.croppad(data_dict, metadata)
		return self.composed_transforms(data_dict)



class YuccaTestDataset(Dataset):
	def __init__(self, raw_data_dir, patch_size):
		self.data_path = raw_data_dir
		self.unique_cases = np.unique([i[:-len('_000.nii.gz')] 
								 for i in subfiles(self.data_path, suffix='.nii.gz', join=False)])
		self.patch_size = patch_size

	def __len__(self):
		return len(self.unique_cases)

	def __getitem__(self, idx):
		# Here we generate the paths to the cases along with their ID which they will be saved as.
		# we pass "case" as a list of strings and case_id as a string to the dataloader which 
		# will convert them to a list of tuples of strings and a tuple of a string.
		# i.e. ['path1', 'path2'] -> [('path1',), ('path2',)]
		case_id = self.unique_cases[idx]
		case = [impath for impath in subfiles(self.data_path, suffix='.nii.gz') 
		  if os.path.split(impath)[-1][:-len('_000.nii.gz')] == case_id]
		return case, case_id