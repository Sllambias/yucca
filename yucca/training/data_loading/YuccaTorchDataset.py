import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join
from torch.utils.data import Dataset, IterableDataset
from yucca.training.augmentation.YuccaAugmenter import YuccaAugmenter
import random

class YuccaDataset(Dataset):
    def __init__(self, path_to_preprocessed_data, augmentation_parameters):
        self.data_path = path_to_preprocessed_data
        self.files = subfiles(self.data_path, suffix = '.npz', join = False)
        self.transform = YuccaAugmenter(augmentation_parameters)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = self.files[idx]
        data = np.load(join(self.data_path, data))['data']
        data = {'image': data[:-1], 'seg': data[-1:]}
        return self.transform(**data)