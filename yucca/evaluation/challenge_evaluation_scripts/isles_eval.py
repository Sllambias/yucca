__author__ = "Ezequiel de la Rosa"
import numpy as np
import warnings
import cc3d
import argparse
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json


def compute_dice(im1, im2, empty_value=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size as im1. If not boolean, it will be converted.
    empty_value : scalar, float.

    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        If both images are empty (sum equal to zero) = empty_value

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.

    This function has been adapted from the Verse Challenge repository:
    https://github.com/anjany/verse/blob/main/utils/eval_utilities.py
    """

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_value

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2.0 * intersection.sum() / im_sum


def compute_absolute_volume_difference(im1, im2, voxel_size):
    """
    Computes the absolute volume difference between two masks.

    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    voxel_size : scalar, float (ml)
        If not float, it will be converted.

    Returns
    -------
    abs_vol_diff : float, measured in ml.
        Absolute volume difference as a float.
        Maximum similarity = 0
        No similarity = inf


    Notes
    -----
    The order of inputs is irrelevant. The result will be identical if `im1` and `im2` are switched.
    """

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    voxel_size = voxel_size.astype(np.float)

    if im1.shape != im2.shape:
        warnings.warn(
            "Shape mismatch: ground_truth and prediction have difference shapes."
            " The absolute volume difference is computed with mismatching shape masks"
        )

    ground_truth_volume = np.sum(im1) * voxel_size
    prediction_volume = np.sum(im2) * voxel_size
    abs_vol_diff = np.abs(ground_truth_volume - prediction_volume)

    return abs_vol_diff


def compute_absolute_lesion_difference(ground_truth, prediction, connectivity=26):
    """
    Computes the absolute lesion difference between two masks. The number of lesions are counted for
    each volume, and their absolute difference is computed.

    Parameters
    ----------
    ground_truth : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    prediction : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.

    Returns
    -------
    abs_les_diff : int
        Absolute lesion difference as integer.
        Maximum similarity = 0
        No similarity = inf


    Notes
    -----
    """
    ground_truth = np.asarray(ground_truth).astype(bool)
    prediction = np.asarray(prediction).astype(bool)

    _, ground_truth_numb_lesion = cc3d.connected_components(ground_truth, connectivity=connectivity, return_N=True)
    _, prediction_numb_lesion = cc3d.connected_components(prediction, connectivity=connectivity, return_N=True)
    abs_les_diff = abs(ground_truth_numb_lesion - prediction_numb_lesion)
    print(ground_truth_numb_lesion, prediction_numb_lesion)

    return abs_les_diff


def compute_lesion_f1_score(ground_truth, prediction, empty_value=1.0, connectivity=26):
    """
    Computes the lesion-wise F1-score between two masks.

    Parameters
    ----------
    ground_truth : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    prediction : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    empty_value : scalar, float.
    connectivity : scalar, int.

    Returns
    -------
    f1_score : float
        Lesion-wise F1-score as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value

    Notes
    -----
    This function computes lesion-wise score by defining true positive lesions (tp), false positive lesions (fp) and
    false negative lesions (fn) using 3D connected-component-analysis.

    tp: 3D connected-component from the ground-truth image that overlaps at least on one voxel with the prediction image.
    fp: 3D connected-component from the prediction image that has no voxel overlapping with the ground-truth image.
    fn: 3d connected-component from the ground-truth image that has no voxel overlapping with the prediction image.
    """
    ground_truth = np.asarray(ground_truth).astype(bool)
    prediction = np.asarray(prediction).astype(bool)
    tp = 0
    fp = 0
    fn = 0

    # Check if ground-truth connected-components are detected or missed (tp and fn respectively).
    intersection = np.logical_and(ground_truth, prediction)
    labeled_ground_truth, N = cc3d.connected_components(ground_truth, connectivity=connectivity, return_N=True)

    # Iterate over ground_truth clusters to find tp and fn.
    # tp and fn are only computed if the ground-truth is not empty.
    if N > 0:
        for _, binary_cluster_image in cc3d.each(labeled_ground_truth, binary=True, in_place=True):
            if np.logical_and(binary_cluster_image, intersection).any():
                tp += 1
            else:
                fn += 1

    # iterate over prediction clusters to find fp.
    # fp are only computed if the prediction image is not empty.
    labeled_prediction, N = cc3d.connected_components(prediction, connectivity=connectivity, return_N=True)
    if N > 0:
        for _, binary_cluster_image in cc3d.each(labeled_prediction, binary=True, in_place=True):
            if not np.logical_and(binary_cluster_image, ground_truth).any():
                fp += 1

    # Define case when both images are empty.
    if tp + fp + fn == 0:
        _, N = cc3d.connected_components(ground_truth, connectivity=connectivity, return_N=True)
        if N == 0:
            f1_score = empty_value
    else:
        f1_score = tp / (tp + (fp + fn) / 2)

    return f1_score


import nibabel as nib

gts = [
    "/home/zcr545/YuccaData/yucca_raw_data/Task049_3BrainLesion3Labels/labelsTs/3BL3L_1.nii.gz",
    "/home/zcr545/YuccaData/yucca_raw_data/Task049_3BrainLesion3Labels/labelsTs/3BL3L_sub-strokecase0049.nii.gz",
    "/home/zcr545/YuccaData/yucca_raw_data/Task049_3BrainLesion3Labels/labelsTs/3BL3L_UNC_train_Case09.nii.gz",
]
preds = [
    "/home/zcr545/YuccaData/yucca_segmentations/Task049_3BrainLesion3Labels/Task049_3BrainLesion3Labels/UNet2D/YuccaTrainerV2__YuccaPlannerV2_Ensemble/fold_0_checkpoint_best/3BL3L_1.nii.gz",
    "/home/zcr545/YuccaData/yucca_segmentations/Task049_3BrainLesion3Labels/Task049_3BrainLesion3Labels/UNet2D/YuccaTrainerV2__YuccaPlannerV2_Ensemble/fold_0_checkpoint_best/3BL3L_sub-strokecase0049.nii.gz",
    "/home/zcr545/YuccaData/yucca_segmentations/Task049_3BrainLesion3Labels/Task049_3BrainLesion3Labels/UNet2D/YuccaTrainerV2__YuccaPlannerV2_Ensemble/fold_0_checkpoint_best/3BL3L_UNC_train_Case09.nii.gz",
]
fp = 0
tp = 0
fn = 0
tplog = []
fnlog = []
labelarr = np.array([1, 2, 3])
for i in range(1):
    gt = nib.load(gts[i]).get_fdata()
    pred = nib.load(preds[i]).get_fdata()
    for label in labelarr:
        pred_for_label = np.where(pred == label, pred, 0).astype(bool)
        gt_for_label = np.where(gt == label, gt, 0).astype(bool)
        intersection = np.logical_and(gt_for_label, pred_for_label)
        gt_cc, gt_cc_n = cc3d.connected_components(gt_for_label, connectivity=26, return_N=True)
        pred_cc, pred_cc_n = cc3d.connected_components(pred_for_label, connectivity=26, return_N=True)
        # Compute Dice coefficient
        if gt_cc_n > 0:
            for _, binary_cluster_image in cc3d.each(gt_cc, binary=True, in_place=True):
                if np.logical_and(binary_cluster_image, pred_cc).any():
                    tp += 1
                    tplog.append(binary_cluster_image.sum())
                else:
                    fn += 1
                    fnlog.append(binary_cluster_image.sum())
        if pred_cc_n > 0:
            for _, binary_cluster_image in cc3d.each(pred_cc, binary=True, in_place=True):
                if not np.logical_and(binary_cluster_image, gt_cc).any():
                    fp += 1
print(tp, fp, fn, gt_cc_n, pred_cc_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    args = parser.parse_args()

    pred = args.pred
    gt = args.gt

    predfiles = subfiles(pred, join=False, suffix=".nii.gz")
    metrics = {}
    all_dice = []
    all_abs = []
    all_f1 = []
    for sub in predfiles:
        metrics[sub] = {}
        im1 = nib.load(join(gt, sub)).get_fdata()
        im2 = nib.load(join(pred, sub)).get_fdata()

        dice = compute_dice(im1, im2)
        abs_les_diff = compute_absolute_lesion_difference(im1, im2)
        les_f1 = compute_lesion_f1_score(im1, im2)
        metrics[sub]["dice"] = dice
        all_dice.append(dice)
        metrics[sub]["absolute lesion difference"] = abs_les_diff
        all_abs.append(abs_les_diff)
        metrics[sub]["lesion f1 score"] = les_f1
        all_f1.append(les_f1)

    metrics["mean"] = {}
    metrics["mean"]["dice"] = float(np.mean(all_dice))
    metrics["mean"]["absolute lesion difference"] = float(np.mean(all_abs))
    metrics["mean"]["lesion f1 score"] = float(np.mean(all_f1))
    save_json(metrics, join(pred, "results_isles_eval.json"))
