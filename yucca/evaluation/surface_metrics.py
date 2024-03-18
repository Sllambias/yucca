import numpy as np
from yucca.utils.nib_utils import get_nib_spacing
from yucca.evaluation.deepmind_surface_distance import metrics


def get_surface_metrics_for_label(gt, pred, label, as_binary: bool = False):
    spacing = get_nib_spacing(pred)
    pred = pred.get_fdata()
    gt = gt.get_fdata()
    labeldict = {}

    if label == 0:
        labeldict["Average Surface Distance"] = 0
        return labeldict
    if as_binary:
        gt = gt.astype(bool)
        pred = pred.astype(bool)
    else:
        pred = np.where(pred == label, 1, 0).astype(bool)
        gt = np.where(gt == label, 1, 0).astype(bool)

    surface_distances = metrics.compute_surface_distances(
        mask_gt=gt,
        mask_pred=pred,
        spacing_mm=spacing,
    )

    labeldict["Average Surface Distance"] = metrics.compute_surface_dice_at_tolerance(
        surface_distances=surface_distances, tolerance_mm=1
    )
    return labeldict
