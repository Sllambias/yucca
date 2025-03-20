import numpy as np
import torch
import os
from typing import Union, Optional
from batchgenerators.utilities.file_and_folder_operations import subfiles
from yucca.modules.data.augmentation.transforms.cropping_and_padding import CropPad
from yucca.modules.data.augmentation.transforms.formatting import NumpyToTorch
from yucca.modules.data.datasets.YuccaDataset import YuccaTrainDataset, YuccaTestDataset


class ClassificationTrainDataset(YuccaTrainDataset):
    def __init__(
        self,
        samples: list,
        patch_size: list | tuple,
        keep_in_ram: Union[bool, None] = None,
        label_dtype: Optional[Union[int, float]] = torch.int32,
        task_type: str = "classification",
        composed_transforms=None,
        allow_missing_modalities=False,
        p_oversample_foreground=0.33,
    ):
        self.all_cases = samples
        self.composed_transforms = composed_transforms
        self.patch_size = patch_size
        self.label_dtype = label_dtype
        self.allow_missing_modalities = allow_missing_modalities
        self.already_loaded_cases = {}

        self.croppad = CropPad(patch_size=self.patch_size, label_key=None, p_oversample_foreground=p_oversample_foreground)
        self.to_torch = NumpyToTorch(label_dtype=self.label_dtype)

        self._keep_in_ram = keep_in_ram

    def __getitem__(self, idx):
        # remove extension if file splits include extensions
        case, _ = os.path.splitext(self.all_cases[idx])
        data = self.load_and_maybe_keep_volume(case)
        metadata = self.load_and_maybe_keep_pickle(case)

        if self.allow_missing_modalities:
            image, label = self.unpack_with_zeros(data)
        else:
            image, label = self.unpack(data)

        data_dict = {"file_path": case}
        data_dict.update({"image": image, "label": label})

        return self._transform(data_dict, metadata)

    def unpack(self, data):
        return data[0], data[-1][0]

    def unpack_with_zeros(self, data):
        assert data.dtype == "object", "allow missing modalities is true but dtype is not object"

        # First find the array with the largest array.
        # in classification this avoids setting the zero array to the 1d array with classes
        sizes = [i.size for i in data]
        idx_largest_array = np.where(sizes == np.max(sizes))[0][0]

        # replace missing modalities with zero-filed arrays
        for idx, i in enumerate(data):
            if i.size == 0:
                data[idx] = np.zeros(data[idx_largest_array].squeeze().shape)

        # unpack array into images and labels
        images = np.array([mod for mod in data[:-1]])
        label = data[-1:][0]

        return images, label


class ClassificationTestDataset(YuccaTestDataset):
    def __init__(
        self,
        raw_data_dir: str,
        pred_save_dir: str,
        overwrite_predictions: bool = False,
        suffix="nii.gz",
        prediction_suffix=None,
        pred_include_cases: list = None,
    ):
        super().__init__(
            raw_data_dir=raw_data_dir,
            pred_save_dir=pred_save_dir,
            overwrite_predictions=overwrite_predictions,
            suffix=suffix,
            prediction_suffix="txt",
            pred_include_cases=pred_include_cases,
        )


class ClassificationTrainDatasetWithCovariates(ClassificationTrainDataset):
    def __init__(
        self,
        samples: list,
        patch_size: list | tuple,
        keep_in_ram: Union[bool, None] = None,
        label_dtype: Optional[Union[int, float]] = torch.int32,
        task_type: str = "classification",
        composed_transforms=None,
        allow_missing_modalities=False,
        p_oversample_foreground=0.33,
    ):
        super().__init__(
            samples=samples,
            patch_size=patch_size,
            keep_in_ram=keep_in_ram,
            label_dtype=label_dtype,
            task_type=task_type,
            composed_transforms=composed_transforms,
            allow_missing_modalities=allow_missing_modalities,
            p_oversample_foreground=p_oversample_foreground,
        )

    def __getitem__(self, idx):
        # remove extension if file splits include extensions
        case, _ = os.path.splitext(self.all_cases[idx])
        data = self.load_and_maybe_keep_volume(case)
        metadata = self.load_and_maybe_keep_pickle(case)

        image, covariates, label = self.unpack(data)
        data_dict = {"file_path": case}
        data_dict.update({"image": image, "covariates": covariates, "label": label})

        return self._transform(data_dict, metadata)

    def unpack(self, data):
        return data[0], data[-2], data[-1][0]


class ClassificationTestDatasetWithCovariates(YuccaTestDataset):
    def __init__(
        self,
        raw_data_dir: str,
        pred_save_dir: str,
        overwrite_predictions: bool = False,
        suffix="nii.gz",
        prediction_suffix=None,
        pred_include_cases: list = None,
    ):
        super().__init__(
            raw_data_dir=raw_data_dir,
            pred_save_dir=pred_save_dir,
            overwrite_predictions=overwrite_predictions,
            suffix=suffix,
            prediction_suffix="txt",
            pred_include_cases=pred_include_cases,
        )

    def __getitem__(self, idx):
        # Here we generate the paths to the cases along with their ID which they will be saved as.
        # we pass "case" as a list of strings and case_id as a string to the dataloader which
        # will convert them to a list of tuples of strings and a tuple of a string.
        # i.e. ['path1', 'path2'] -> [('path1',), ('path2',)]
        case_id = self.unique_cases[idx]
        image_paths = [
            impath
            for impath in subfiles(self.data_path, suffix=self.suffix)
            if os.path.split(impath)[-1][: -len("_000." + self.suffix)] == case_id
        ]
        covariatepath = self.data_path.replace("imagesTs", "covariatesTs")
        covariates = torch.tensor(np.loadtxt(os.path.join(covariatepath, case_id + "_COV.txt"))).unsqueeze(0)
        return {"data_paths": image_paths, "covariates": covariates, "extension": self.suffix, "case_id": case_id}


if __name__ == "__main__":
    from batchgenerators.utilities.file_and_folder_operations import subfiles

    files = subfiles("/home/zcr545/yuccadata/yucca_preprocessed/Task503_ADNI300_MRI/ClassificationV2_112x224x224")
    data = ClassificationDataset(files, patch_size=(12, 12, 12))
