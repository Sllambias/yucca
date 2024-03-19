import numpy as np
import torch
import os
from typing import Union, Literal, Optional
import logging
from batchgenerators.utilities.file_and_folder_operations import subfiles, load_pickle
from yucca.image_processing.transforms.cropping_and_padding import CropPad
from yucca.image_processing.transforms.formatting import NumpyToTorch


class YuccaTrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: list,
        patch_size: list | tuple,
        keep_in_ram: Union[bool, None] = None,
        label_dtype: Optional[Union[int, float]] = None,
        composed_transforms=None,
        task_type: Literal["classification", "segmentation", "self-supervised", "contrastive"] = "segmentation",
    ):
        self.all_cases = samples
        self.composed_transforms = composed_transforms
        self.patch_size = patch_size
        self.task_type = task_type
        assert task_type in ["classification", "segmentation", "self-supervised", "contrastive"]
        self.label_dtype = label_dtype

        self.already_loaded_cases = {}

        # for segmentation and classification we override the default None
        # because arrays are saved as floats and we want them to be ints.
        if self.label_dtype is None:
            if self.task_type in ["segmentation", "classification"]:
                self.label_dtype = torch.int32

        self.croppad = CropPad(patch_size=self.patch_size, p_oversample_foreground=0.33)
        self.to_torch = NumpyToTorch(label_dtype=self.label_dtype)

        self._keep_in_ram = keep_in_ram

    @property
    def keep_in_ram(self):
        if self._keep_in_ram is not None:
            return self._keep_in_ram
        if len(self.all_cases) < 50:
            self._keep_in_ram = True
        else:
            logging.debug("Large dataset detected. Will not keep cases in RAM during training.")
            self._keep_in_ram = False
        return self._keep_in_ram

    def load_and_maybe_keep_pickle(self, picklepath):
        if not self.keep_in_ram:
            return load_pickle(picklepath)
        if picklepath in self.already_loaded_cases:
            return self.already_loaded_cases[picklepath]
        self.already_loaded_cases[picklepath] = load_pickle(picklepath)
        return self.already_loaded_cases[picklepath]

    def load_and_maybe_keep_volume(self, path):
        if not self.keep_in_ram:
            if path[-3:] == "npy":
                try:
                    return np.load(path, "r")
                except ValueError:
                    return np.load(path, allow_pickle=True)
            image = np.load(path)
            assert len(image.files) == 1, (
                "More than one entry in data array. " f"Should only be ['data'] but is {[key for key in image.files]}"
            )
            return image[image.files[0]]

        if path in self.already_loaded_cases:
            return self.already_loaded_cases[path]

        if path[-3:] == "npy":
            try:
                self.already_loaded_cases[path] = np.load(path, "r")
            except ValueError:
                self.already_loaded_cases[path] = np.load(path, allow_pickle=True)
            return self.already_loaded_cases[path]

        image = np.load(path)
        assert len(image.files) == 1, (
            "More than one entry in data array. " f"Should only be ['data'] but is {[key for key in image.files]}"
        )
        self.already_loaded_cases = image[image.files[0]]
        return self.already_loaded_cases[path]

    def __len__(self):
        return len(self.all_cases)

    def __getitem__(self, idx):
        case = self.all_cases[idx]
        data = self.load_and_maybe_keep_volume(case)
        data_dict = {"file_path": case}  # metadata that can be very useful for debugging.
        if self.task_type == "classification":
            data_dict.update({"image": data[:-1][0], "label": data[-1:][0]})
        elif self.task_type == "segmentation":
            data_dict.update({"image": data[:-1], "label": data[-1:]})
        elif self.task_type == "self-supervised":
            data_dict.update({"image": data})
        elif self.task_type == "contrastive":
            view1 = self._transform({"image": data}, case)["image"]
            view2 = self._transform({"image": data}, case)["image"]
            data_dict.update({"image": (view1, view2)})
            return data_dict
        else:
            logging.error(f"Task Type not recognized. Found {self.task_type}")
        return self._transform(data_dict, case)

    def _transform(self, data_dict, case):
        metadata = self.load_and_maybe_keep_pickle(case[: -len(".npy")] + ".pkl")
        data_dict = self.croppad(data_dict, metadata)
        if self.composed_transforms is not None:
            data_dict = self.composed_transforms(data_dict)
        return self.to_torch(data_dict)


class YuccaTestDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data_dir: str, pred_save_dir: str, overwrite_predictions: bool = False, suffix="nii.gz"):
        self.data_path = raw_data_dir
        self.pred_save_dir = pred_save_dir
        self.overwrite = overwrite_predictions
        self.suffix = suffix
        self.unique_cases = np.unique(
            [i[: -len("_000." + suffix)] for i in subfiles(self.data_path, suffix=self.suffix, join=False)]
        )
        assert len(self.unique_cases) > 0, f"No cases found in {self.data_path}. Looking for files with suffix: {self.suffix}"

        self.cases_already_predicted = np.unique(
            [i[: -len("." + suffix)] for i in subfiles(self.pred_save_dir, suffix=self.suffix, join=False)]
        )
        logging.info(f"Found {len(self.cases_already_predicted)} already predicted cases. Overwrite: {self.overwrite}")
        if not self.overwrite:
            self.unique_cases = [i for i in self.unique_cases if i not in self.cases_already_predicted]

    def __len__(self):
        return len(self.unique_cases)

    def __getitem__(self, idx):
        # Here we generate the paths to the cases along with their ID which they will be saved as.
        # we pass "case" as a list of strings and case_id as a string to the dataloader which
        # will convert them to a list of tuples of strings and a tuple of a string.
        # i.e. ['path1', 'path2'] -> [('path1',), ('path2',)]
        case_id = self.unique_cases[idx]
        case = [
            impath
            for impath in subfiles(self.data_path, suffix=self.suffix)
            if os.path.split(impath)[-1][: -len("_000." + self.suffix)] == case_id
        ]
        return case, case_id


if __name__ == "__main__":
    import torch
    from yucca.paths import yucca_preprocessed_data
    from batchgenerators.utilities.file_and_folder_operations import join
    from yucca.training.data_loading.samplers import InfiniteRandomSampler

    files = subfiles(join(yucca_preprocessed_data, "Task001_OASIS/YuccaPlanner"), suffix="npy")
    ds = YuccaTrainDataset(files, patch_size=(12, 12, 12))
    sampler = InfiniteRandomSampler(ds)
    dl = torch.utils.data.DataLoader(ds, num_workers=2, batch_size=2, sampler=sampler)
