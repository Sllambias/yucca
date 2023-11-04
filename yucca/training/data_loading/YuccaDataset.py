#%%
import numpy as np
import torch
import os
from batchgenerators.utilities.file_and_folder_operations import subfiles, load_pickle
from yuccalib.image_processing.transforms.cropping_and_padding import CropPad


class YuccaTrainDataset(torch.utils.data.Dataset):
    def __init__(self,
                 preprocessed_data_dir: list,
                 patch_size: list | tuple,
                 keep_in_ram=False,
                 composed_transforms=None):
        self.all_cases = preprocessed_data_dir
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
        data = self.load_and_maybe_keep_volume(case)
        metadata = self.load_and_maybe_keep_pickle(case[:-len('.npy')] + '.pkl')
        data_dict = {'image': data[:-1], 'seg': data[-1:]}
        data_dict = self.croppad(data_dict, metadata)
        return self.composed_transforms(data_dict)


class YuccaTrainIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 preprocessed_data_dir: list,
                 patch_size: list | tuple,
                 keep_in_ram=False,
                 composed_transforms=None,
                 return_indices: bool = False):
        self.all_cases = preprocessed_data_dir
        self.keep_in_ram = keep_in_ram
        self.croppad = CropPad(patch_size=patch_size, p_oversample_foreground=0.33)
        self.composed_transforms = composed_transforms
        self.already_loaded_cases = {}
        self.start = 0
        self.num_samples = len(self.all_cases)
        self.return_indices = return_indices    # Mainly useful for debugging

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

    def worker_fn(self, indices):
        for idx in indices:
            case = self.all_cases[idx]
            data = self.load_and_maybe_keep_volume(case)
            metadata = self.load_and_maybe_keep_pickle(case[:-len('.npy')] + '.pkl')
            data_dict = {'image': data[:-1], 'seg': data[-1:]}
            data_dict = self.croppad(data_dict, metadata)
            if self.return_indices:
                yield idx
            else:
                if self.composed_transforms:
                    yield self.composed_transforms(data_dict)
                yield data_dict

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.num_samples
        else:  # in a worker process
            # split workload
            per_worker = int(np.ceil((self.num_samples - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.num_samples)
        #yield worker_id, np.random.choice(list(range(iter_start, iter_end)), 4, replace=True)
        yield from self.worker_fn(list(range(iter_start, iter_end)))


class YuccaTestDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data_dir, patch_size):
        self.data_path = raw_data_dir
        self.unique_cases = np.unique([i[:-len('_000.nii.gz')]
                                       for i in subfiles(self.data_path, suffix='.nii.gz',
                                                         join=False)])
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

#%%
files = subfiles('/home/zcr545/YuccaData/yucca_preprocessed/Task001_OASIS/YuccaPlanner', suffix='npy')
ds = YuccaTrainIterableDataset(files, patch_size=(12, 12, 12), return_indices=True)

dl = torch.utils.data.DataLoader(ds, num_workers=2, batch_size=4)
#for i in range(10):
for i in dl:
    print(i)
 #   print(next(iter(dl)))
    #print(list(dl))
    #print(list(dl)[0]['image'].shape)
# %%
