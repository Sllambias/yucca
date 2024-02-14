import numpy as np
from scipy.spatial.distance import directed_hausdorff
import numpy.typing as npt

def dice(tp, fp, tn, fn):
    try:
        return (2 * tp) / (2 * tp + fp + fn)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def dice_per_label(tp_list, fp_list, tn_list, fn_list):
    return [dice(tp_list[i], fp_list[i], tn_list[i], fn_list[i]) for i in range(len(tp_list))]

def jaccard(tp, fp, tn, fn):
    try:
        return (tp) / (tp + fp + fn)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def jaccard_per_label(tp_list, fp_list, tn_list, fn_list):
    return [dice(tp_list[i], fp_list[i], tn_list[i], fn_list[i]) for i in range(len(tp_list))]

def sensitivity(tp, fp, tn, fn):
    # recall, hit rate, tpr
    # How many cases of X are correctly recognized as X?
    try:
        return tp / (tp + fn)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def specificity(tp, fp, tn, fn):
    # beware with using TN-metrics.
    try:
        return tn / (tn + fp)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def precision(tp, fp, tn, fn):
    # When X is predicted, how often is it truly X?
    try:
        return tp / (tp + fp)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def volume_similarity(tp, fp, tn, fn):
    try:
        return 1 - abs(fn - fp) / (2 * tp + fn + fp)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def f1(tp, fp, tn, fn):
    try:
        prec = precision(tp, fp, tn, fn)
        sens = sensitivity(tp, fp, tn, fn)
        return 2 * (prec * sens) / (prec + sens)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan

def hausdorff_distance(y_true: npt.ArrayLike, y_score: npt.ArrayLike):
    
    assert len(y_true.shape) == 1 or y_true.shape == y_score.shape, "y_true must be 1D or 2D"
    hauss = directed_hausdorff(y_true, y_score, seed=0)
    return np.sum(hauss)/np.size(hauss)

def TP(tp, fp, tn, fn):
    return tp


def FP(tp, fp, tn, fn):
    return fp


def FN(tp, fp, tn, fn):
    return fn


def total_pos_gt(tp, fp, tn, fn):
    return tp + fn


def total_pos_pred(tp, fp, tn, fn):
    return tp + fp
