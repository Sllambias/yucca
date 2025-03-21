import nibabel as nib
import numpy as np
from yucca.functional.utils.softmax import softmax
from yucca.functional.utils.nib_utils import reorient_nib_image
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    subfiles,
    maybe_mkdir_p as ensure_dir_exists,
)


def save_nifti_from_numpy(pred, outpath, properties, compression=9):
    # slight hacky, but it is what it is
    nib.openers.Opener.default_compresslevel = compression
    pred = nib.Nifti1Image(
        pred,
        properties["nifti_metadata"]["affine"],
        header=nib.Nifti1Header(properties["nifti_metadata"]["header"]),
        dtype=np.uint8,
    )
    if properties["nifti_metadata"]["reoriented"]:
        pred = reorient_nib_image(
            pred, properties["nifti_metadata"]["final_direction"], properties["nifti_metadata"]["original_orientation"]
        )
    nib.save(
        pred,
        outpath + ".nii.gz",
    )
    del pred


def save_png_from_numpy(pred, outpath):
    pred = Image.fromarray(pred.astype(np.uint8))
    pred.save(outpath + ".png")
    del pred


def save_txt_from_numpy(pred, outpath):
    np.savetxt(outpath + ".txt", np.atleast_1d(pred), fmt="%i", delimiter=",")
    del pred


def save_prediction_from_logits(logits, outpath, properties, save_softmax=False, compression=9):
    if save_softmax:
        softmax_result = softmax(logits)[0].astype(np.float32)
        np.savez_compressed(outpath + ".npz", data=softmax_result, properties=properties)
    if logits.shape[1] > 1:
        logits = np.argmax(logits, 1)
    pred = np.squeeze(logits)
    if properties.get("save_format") == "png":
        save_png_from_numpy(pred, outpath)
    elif properties.get("save_format") == "txt":
        save_txt_from_numpy(pred, outpath)
    else:
        save_nifti_from_numpy(pred, outpath, properties, compression=compression)


def save_multilabel_prediction_from_logits(logits, outpath, properties, compression=9):
    nib.openers.Opener.default_compresslevel = compression
    # here we use a sigmoid instead of the softmax/argmax functions to keep the multiple channels
    # of predicted classes. This means predictions from this are stored as (c, h, w, d) rather than (h, w, d)
    pred = 1.0 / (1.0 + np.exp(-logits))
    pred = (pred > 0.5).astype(np.uint8)
    pred = np.squeeze(pred)
    preds = []
    for i in range(pred.shape[0]):
        pred_for_label = nib.Nifti1Image(
            pred[i],
            properties["nifti_metadata"]["affine"],
            header=nib.Nifti1Header(properties["nifti_metadata"]["header"]),
            dtype=np.uint8,
        )
        if properties["nifti_metadata"]["reoriented"]:
            pred_for_label = reorient_nib_image(
                pred_for_label,
                properties["nifti_metadata"]["final_direction"],
                properties["nifti_metadata"]["original_orientation"],
            )
        preds.append(pred_for_label)
    pred = nib.concat_images(preds)
    nib.save(
        pred,
        outpath + ".nii.gz",
    )
    del pred


def merge_softmax_from_folders(folders: list, outpath, method="sum"):
    ensure_dir_exists(outpath)
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
