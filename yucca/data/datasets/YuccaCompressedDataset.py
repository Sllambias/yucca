import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile
from yucca.data.augmentation.transforms.formatting import NumpyToTorch
from yucca.data.datasets.YuccaDataset import YuccaTrainDataset


class YuccaCompressedTrainDataset(YuccaTrainDataset):
    def load_and_maybe_keep_volume(self, path: str):
        path = path + ".npz"
        if isfile(path):
            try:
                return np.load(path, "r")["data"]
            except ValueError:
                return np.load(path, allow_pickle=True)["data"]
        else:
            print("compressed data not found.")
