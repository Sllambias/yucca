import numpy as np
import cc3d
from yucca.utils.nib_utils import get_nib_spacing
from yucca.evaluation.metrics import sensitivity, precision, f1


def get_obj_stats_for_label(gt, pred, label, as_binary=False):
    spacing = get_nib_spacing(gt)
    gt = np.around(gt.get_fdata())
    pred = np.around(pred.get_fdata())
    if as_binary:
        gt = gt.astype(bool).astype(int)
        pred = pred.astype(bool).astype(int)
    labeldict = {}
    if label == 0:
        labeldict["_OBJ Total Objects Prediction"] = 0
        labeldict["_OBJ Total Objects Ground Truth"] = 0
        labeldict["_OBJ True Positives"] = 0
        labeldict["_OBJ False Positives"] = 0
        labeldict["_OBJ False Negatives"] = 0
        labeldict["_OBJ Mean Volume Prediction"] = 0
        labeldict["_OBJ Mean Volume Ground Truth"] = 0
        labeldict["_OBJ sensitivity"] = 0
        labeldict["_OBJ precision"] = 0
        labeldict["_OBJ F1"] = 0
    else:
        pred_for_label = np.where(pred == label, 1, 0).astype(bool)
        gt_for_label = np.where(gt == label, 1, 0).astype(bool)
        cc_gt, cc_gt_n = cc3d.connected_components(gt_for_label, connectivity=26, return_N=True)
        cc_pred, cc_pred_n = cc3d.connected_components(pred_for_label, connectivity=26, return_N=True)
        tp, fp, fn, gtvols, predvols = obj_get_tp_fp_fn_gtvols_predvols(cc_gt, cc_pred, cc_gt_n, cc_pred_n)

        labeldict["_OBJ Total Objects Prediction"] = cc_pred_n
        labeldict["_OBJ Total Objects Ground Truth"] = cc_gt_n
        labeldict["_OBJ True Positives"] = tp
        labeldict["_OBJ False Positives"] = fp
        labeldict["_OBJ False Negatives"] = fn
        labeldict["_OBJ Mean Volume Prediction"] = float(np.prod(spacing) * np.mean(predvols))
        labeldict["_OBJ Mean Volume Ground Truth"] = float(np.prod(spacing) * np.mean(gtvols))
        labeldict["_OBJ sensitivity"] = sensitivity(tp, fp, 0, fn)
        labeldict["_OBJ precision"] = precision(tp, fp, 0, fn)
        labeldict["_OBJ F1"] = f1(tp, fp, 0, fn)
    return labeldict


def obj_get_tp_fp_fn_gtvols_predvols(cc_gt, cc_pred, n_cc_gt, n_cc_pred):
    tp = 0
    fp = 0
    fn = 0
    predvols = []
    gtvols = []
    if n_cc_gt > 0:
        for _, binary_cluster_image in cc3d.each(cc_gt, binary=True, in_place=True):
            gtvols.append(binary_cluster_image.sum())
            if np.logical_and(binary_cluster_image, cc_pred).any():
                tp += 1
            else:
                fn += 1
    if n_cc_pred > 0:
        for _, binary_cluster_image in cc3d.each(cc_pred, binary=True, in_place=True):
            predvols.append(binary_cluster_image.sum())
            if not np.logical_and(binary_cluster_image, cc_gt).any():
                fp += 1
    if not predvols:
        predvols = 0
    if not gtvols:
        gtvols = 0
    return tp, fp, fn, gtvols, predvols


def n_cc_gt(cc_gt, cc_pred, n_cc_gt, n_cc_pred):  # noqa: U100
    return n_cc_gt


def n_cc_pred(cc_gt, cc_pred, n_cc_gt, n_cc_pred):  # noqa: U100
    return n_cc_pred
