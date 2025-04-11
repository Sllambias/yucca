import torch


class SingleFileTestDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data_dir: str, pred_save_dir: str, suffix=".nii.gz", **kwargs):  # noqa: U100
        self.data_path = raw_data_dir
        self.pred_save_dir = pred_save_dir
        self.suffix = suffix

    def __len__(self):
        return 1

    def __getitem__(self, idx):  # noqa: U100
        # Here we generate the paths to the cases along with their ID which they will be saved as.
        # we pass "case" as a list of strings and case_id as a string to the dataloader which
        # will convert them to a list of tuples of strings and a tuple of a string.
        # i.e. ['path1', 'path2'] -> [('path1',), ('path2',)]

        return {"data_paths": [self.data_path], "extension": self.suffix, "case_id": ""}
