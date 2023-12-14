import nibabel as nib
import numpy as np
from lightning.pytorch.callbacks import BasePredictionWriter
from yucca.utils.softmax import softmax
from yucca.utils.nib_utils import reorient_nib_image
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    subfiles,
    subdirs,
    maybe_mkdir_p,
)


def save_nifti_from_numpy(pred, outpath, properties, compression=9):
    # slight hacky, but it is what it is
    nib.openers.Opener.default_compresslevel = compression
    pred = nib.Nifti1Image(pred, properties["affine"], dtype=np.uint8)
    if properties["reoriented"]:
        pred = reorient_nib_image(pred, properties["new_orientation"], properties["original_orientation"])
    pred.set_qform(properties["qform"])
    pred.set_sform(properties["sform"])
    nib.save(
        pred,
        outpath + ".nii.gz",
    )
    del pred


def save_png_from_numpy(pred, outpath, properties, compression=9):
    pred = Image.fromarray(pred)
    pred.save(outpath)
    del pred


def save_txt_from_numpy(pred, outpath, properties):
    np.savetxt(outpath, np.atleast_1d(pred), fmt="%i", delimiter=",")
    del pred


def save_prediction_from_logits(logits, outpath, properties, save_softmax=False, compression=9):
    if save_softmax:
        softmax_result = softmax(logits)[0].astype(np.float32)
        np.savez_compressed(outpath + ".npz", data=softmax_result, properties=properties)
    if logits.shape[1] > 1:
        logits = np.argmax(logits, 1)
    pred = np.squeeze(logits)
    if properties.get("save_format") == "png":
        save_png_from_numpy(pred, outpath, properties)
    if properties.get("save_format") == "txt":
        save_txt_from_numpy(pred, outpath, properties)
    else:
        save_nifti_from_numpy(pred, outpath, properties, compression=compression)


def merge_softmax_from_folders(folders: list, outpath, method="sum"):
    maybe_mkdir_p(outpath)
    cases = subfiles(folders[0], suffix=".npz", join=False)
    for folder in folders:
        assert cases == subfiles(folder, suffix=".npz", join=False), (
            f"Found unexpected cases. "
            f"The following two folders do not contain the same cases: \n"
            f"{folders[0]} \n"
            f"{folder}"
        )

    for case in cases:
        files_for_case = [np.load(join(folder, case), allow_pickle=True) for folder in folders]
        properties_for_case = files_for_case[0]["properties"]
        files_for_case = [file["data"].astype(np.float32) for file in files_for_case]

        if method == "sum":
            files_for_case = np.sum(files_for_case, axis=0)

        files_for_case = np.argmax(files_for_case, 0)
        save_nifti_from_numpy(
            files_for_case,
            join(outpath, case[:-4]),
            properties=properties_for_case.item(),
        )

    del files_for_case, properties_for_case


class WritePredictionFromLogits(BasePredictionWriter):
    def __init__(self, output_dir, save_softmax, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.save_softmax = save_softmax

    def write_on_batch_end(self, trainer, pl_module, data_dict, batch_indices, *args):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        logits, properties, case_id = (
            data_dict["logits"],
            data_dict["properties"],
            data_dict["case_id"],
        )
        save_prediction_from_logits(
            logits,
            join(self.output_dir, case_id),
            properties=properties,
            save_softmax=self.save_softmax,
        )
