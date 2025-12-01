import numpy as np
import os
from yucca.modules.data.datasets.YuccaDataset import YuccaTrainDataset


class YuccaCompressedTrainDataset(YuccaTrainDataset):
    def load_and_maybe_keep_volume(self, path: str):
        path = path + ".npz"
        if os.path.isfile(path):
            try:
                return np.load(path, "r")["data"]
            except ValueError:
                return np.load(path, allow_pickle=True)["data"]
        else:
            print("compressed data not found.")
